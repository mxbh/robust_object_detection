import torch
from torchvision.ops import nms, box_iou
from detectron2.config.config import configurable
from detectron2.structures import Instances, Boxes
from ..utils.boxutils import box_iou_distance, box_center_distance


class CorrectionModule:
    @configurable
    def __init__(self,
                 num_classes,
                 box_correction=None,
                 label_correction=None,
                 distance_limit=1.0,
                 lower_threshold=0.5,
                 softmax_temp=0.5,
                 mining_threshold=1.0):
                 
        self.num_classes = num_classes
        self.distance_limit = distance_limit
        self.lower_threshold = lower_threshold
        self.softmax_temp = softmax_temp

        if softmax_temp > 0:
            self.weight_fn = lambda x:torch.nn.functional.softmax(torch.logit(x)/self.softmax_temp, dim=0)
        elif softmax_temp == 0:
            self.weight_fn = lambda x: (x == x.max()).float() / (x == x.max()).sum()
        else:
            raise ValueError

        self.mining_threshold = mining_threshold
        self.max_iter = 10
        self.eps = 1

        # box correction
        if box_correction is None or box_correction == '':
            self.box_correction = lambda boxes, scores, targets: targets
        elif box_correction == 'k_boxes':
            self.box_distance = box_iou_distance
            self.box_correction = self.k_boxes
        elif box_correction == 'k_centers':
            self.box_distance = box_center_distance
            self.box_correction = self.k_centers
        else:
            raise ValueError('Unknown box correction method: {}'.format(box_correction))

        # label correction
        if label_correction is None or label_correction == '':
            self.label_correction = lambda boxes, scores, targets: targets
        elif label_correction == 'mining':
            self.label_correction = self.label_mining
        else:
            raise ValueError('Unknown label correction method: {}'.format(label_correction))
        
        self.do_nothing = (box_correction is None) and (label_correction is None)

    @classmethod
    def from_config(cls, cfg):
        assert cfg.MODEL.RETINANET.NUM_CLASSES == cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return {'num_classes': cfg.MODEL.RETINANET.NUM_CLASSES,
                'box_correction': cfg.CORRECTION.BOXES.TYPE,
                'distance_limit': cfg.CORRECTION.BOXES.DISTANCE_LIMIT,
                'lower_threshold': cfg.CORRECTION.BOXES.LOWER_THRESH,
                'softmax_temp': cfg.CORRECTION.BOXES.SOFTMAX_TEMP,
                'label_correction': cfg.CORRECTION.LABELS.TYPE,
                'mining_threshold': cfg.CORRECTION.LABELS.MINING_THRESH,
                }


    def __call__(self, pred_instances, gt_instances):
        '''
        :param pred_instances: list(Instances) where instances has fields pred_boxes, pred_classes_scores.
        :param gt_instances: list(Instances) where instances has fields gt_boxes and gt_classes. (noisy)
        :returns: list(Instances) holding corrected targets.
        '''
        if self.do_nothing:
            return gt_instances
        with torch.no_grad():
            new_target_instances = []

            for pred_instances_img, gt_instances_img in zip(pred_instances, gt_instances):

                pred_boxes = pred_instances_img.pred_boxes.tensor
                pred_classes = pred_instances_img.pred_classes
                pred_scores = pred_instances_img.scores # these are softmaxed over classes

                gt_boxes = gt_instances_img.gt_boxes.tensor
                gt_classes = gt_instances_img.gt_classes


                assert pred_boxes.device == gt_boxes.device
                new_boxes = torch.zeros((0,4), device=pred_boxes.device)
                new_classes = torch.zeros(0, dtype=int, device=pred_classes.device)

                for c in range(self.num_classes):
                    # extract predictions and targets for every class (class specific variables denoted with trailing _c)
                    pred_mask_c = pred_classes == c
                    pred_boxes_c = pred_boxes[pred_mask_c]
                    pred_scores_c = pred_scores[pred_mask_c]

                    gt_mask_c = gt_classes == c
                    gt_boxes_c = gt_boxes[gt_mask_c]

                    corrected_boxes = self.box_correction(boxes=pred_boxes_c,
                                                          scores=pred_scores_c,
                                                          targets=gt_boxes_c)
                    corrected_boxes = self.label_correction(boxes=pred_boxes_c,
                                                            scores=pred_scores_c,
                                                            targets=corrected_boxes)

                    new_boxes = torch.cat([new_boxes, corrected_boxes], dim=0)
                    new_classes = torch.cat([new_classes, 
                                             torch.zeros(corrected_boxes.shape[0],
                                                         dtype=int,
                                                         device=new_classes.device).fill_(c)],
                                             dim=0)

                new_instances = Instances(gt_instances_img.image_size)
                new_instances.gt_boxes = Boxes(new_boxes.detach())
                new_instances.gt_classes = new_classes
                new_target_instances.append(new_instances)

            return new_target_instances

    def k_boxes(self, boxes, scores, targets):
        '''
        Correct boxes by k-means with a novel distance function for boxes.
        :param boxes: torch.tensor (n,4)
        :param scores: torch.tensor (n,) in [0,1]
        :param targets: torch.tensor (m,4)
        :returns: Corrected centroids (torch.tensor (m,4)).
        '''
        K = len(targets)
        if K == 0:
            return targets

        mask = scores >= self.lower_threshold
        if mask.sum() == 0:
            return targets

        scores = scores[mask]
        boxes = boxes[mask]
        centroids = torch.clone(targets)
        
        # get neighborhoods
        distance_matrix = self.box_distance(boxes, centroids)
        neighborhoods = [{'boxes': None, 'scores': None} for k in range(K)]
        assignments = [None for k in range(K)]
        
        for k in range(K):
            mask = distance_matrix[:,k] < self.distance_limit
            neighborhoods[k]['boxes'] = boxes[mask]
            neighborhoods[k]['scores'] = scores[mask]

        # start with iterations
        for iteration in range(self.max_iter):
            for k in range(K):
                if len(neighborhoods[k]['boxes']) == 0:
                    assignments[k] = torch.Tensor() # argmin does not work for an empty tensor
                else:
                    assignments[k] = self.box_distance(neighborhoods[k]['boxes'], centroids).argmin(dim=1)
            
            converged = True
            for k in range(K):
                mask_k = assignments[k] == k 

                if mask_k.sum() > 0:
                    boxes_k = neighborhoods[k]['boxes'][mask_k]
                    scores_k = neighborhoods[k]['scores'][mask_k]
                    weights_k = self.weight_fn(scores_k)

                    # new centroid as weighted mean
                    new_centroid = (weights_k.reshape(-1,1) * boxes_k).sum(dim=0) #/ weights_k.sum() # softmax is normalized

                    # check for convergence
                    if (centroids[k] - new_centroid).max() > self.eps:
                        converged = False

                    # update
                    centroids[k] = new_centroid

                else:
                    # reset or do nothing
                    pass

            if converged:
                break
        
        return centroids

    def k_centers(self, boxes, scores, targets):
        '''
        Correct only box centers by k-means.
        :param boxes: torch.tensor (n,4)
        :param scores: torch.tensor (n,) in [0,1]
        :param targets: torch.tensor (m,4)
        :returns: Corrected centroids (torch.tensor (m,4))
        '''
        K = len(targets)
        if K == 0:
            return targets

        mask = scores >= self.lower_threshold
        if mask.sum() == 0:
            return targets

        scores = scores[mask]
        boxes = boxes[mask]
        centroids = torch.clone(targets)
        
        # get neighborhoods
        distance_matrix = self.box_distance(boxes, centroids)
        neighborhoods = [{'boxes': None, 'scores': None} for k in range(K)]
        assignments = [None for k in range(K)]
        
        for k in range(K):
            mask = distance_matrix[:,k] < self.distance_limit
            neighborhoods[k]['boxes'] = boxes[mask]
            neighborhoods[k]['scores'] = scores[mask]

        # start with iterations
        for iteration in range(self.max_iter):
            for k in range(K):
                if len(neighborhoods[k]['boxes']) == 0:
                    assignments[k] = torch.Tensor() # argmin does not work for an empty tensor
                else:
                    assignments[k] = self.box_distance(neighborhoods[k]['boxes'], centroids).argmin(dim=1)
            
            converged = True
            for k in range(K):
                mask_k = assignments[k] == k 

                if mask_k.sum() > 0:
                    boxes_k = neighborhoods[k]['boxes'][mask_k]
                    scores_k = neighborhoods[k]['scores'][mask_k]
                    weights_k = self.weight_fn(scores_k)

                    # new centroid as weighted mean
                    # new_centroid = (weights_k.reshape(-1,1) * boxes_k).sum(dim=0) #/ weights_k.sum() # softmax is normalized
                    centers_k = 0.5 * torch.stack([boxes_k[:,0] + boxes_k[:,2],
                                                   boxes_k[:,1] + boxes_k[:,3]], dim=1)
                    new_center = (weights_k.reshape(-1,1) * centers_k).sum(dim=0)
                    w,h = centroids[k][2] - centroids[k][0], centroids[k][3] - centroids[k][1]
                    new_centroid = torch.tensor([new_center[0] - w // 2,
                                                 new_center[1] - h // 2,
                                                 new_center[0] + w // 2,
                                                 new_center[1] + h // 2
                                                 ], device=centroids[k].device)

                    # check for convergence
                    if (centroids[k] - new_centroid).max() > self.eps:
                        converged = False

                    # update
                    centroids[k] = new_centroid

                else:
                    # reset or do nothing
                    pass

            if converged:
                break
        
        return centroids


    def label_mining(self, boxes, scores, targets):
        '''
        Add boxes that are predicted with a high confidence and do not/hardly overlap with existing boxes.
        Drop boxes with no sufficiently overlapping and sufficiently confident anchor.
        Note: Results in a box_correction as only targets for a single class are processed at a time.
        :param boxes: torch.tensor (n,4)
        :param scores: torch.tensor (n,) in [0,1]
        :param targets: torch.tensor (m,4)
        :returns: Corrected targets (torch.tensor (m_new,4)).
        '''
        iou_threshold = 0.5

        # mining
        max_number = 64 # maximum number of mined boxes
        mask = scores >= self.mining_threshold

        if mask.sum() != 0:
            mined_boxes = torch.clone(boxes[mask])
            masked_scores = scores[mask]

            keep_indices = nms(boxes=mined_boxes, scores=masked_scores, iou_threshold=iou_threshold)
            mined_boxes = mined_boxes[keep_indices]

            if len(targets) > 0:
                # find the boxes that are not overlapping with target boxes
                iou_matrix = box_iou(mined_boxes, targets)
                mined_boxes = mined_boxes[iou_matrix.max(dim=1).values < iou_threshold]

            mined_boxes = mined_boxes[:max_number] # limit number of mined boxes
        else:
            mined_boxes = torch.empty(0,4, device=targets.device)

        return torch.cat([targets, mined_boxes], dim=0)