import torch
from typing import Dict, List
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet, permute_to_N_HWA_K
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.layers import batched_nms, cat, nonzero_tuple


@META_ARCH_REGISTRY.register()
class CustomRetinaNet(RetinaNet):
    '''
    With train_forward() and compute_losses(), we can separate
    the forward pass of inputs from the loss generation.
    '''
    def train_forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        '''
        Do not use the gt_instances of the inputs here.
        '''
        loss_inputs = {}

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]

        anchors = self.anchor_generator(features)
        pred_logits, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [permute_to_N_HWA_K(x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [permute_to_N_HWA_K(x, 4) for x in pred_anchor_deltas]

        loss_inputs['anchors'] = anchors
        loss_inputs['pred_logits'] = pred_logits
        loss_inputs['pred_anchor_deltas'] = pred_anchor_deltas

        pred_logits = [torch.clone(x).detach() for x in pred_logits] # otherwise it will be overriden
        pred_anchor_deltas = [torch.clone(x).detach() for x in pred_anchor_deltas]

        # if self.training:
        #     assert not torch.jit.is_scripting(), "Not supported"
        #     assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        #     gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        #     losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                results = self.inference(
                    anchors, [torch.clone(x) for x in pred_logits], pred_anchor_deltas, images.image_sizes
                )
                self.visualize_training(batched_inputs, results)

        #     return losses

        # start self.inference: results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            pred_logits_per_image = [x[img_idx] for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            # start self.inference_single_image: results_per_image = self.inference_single_image(anchors, pred_logits_per_image, deltas_per_image, image_size)
            box_cls, box_delta = pred_logits_per_image, deltas_per_image

            boxes_all = []
            scores_all = []
            class_idxs_all = []

            # Iterate over every feature level
            for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
                # (HxWxAxK,)
                predicted_prob = box_cls_i.flatten().sigmoid_()

                # Apply two filtering below to make NMS faster.
                # 1. Keep boxes with confidence score higher than threshold
                keep_idxs = predicted_prob > self.score_thresh_train
                predicted_prob = predicted_prob[keep_idxs]
                topk_idxs = nonzero_tuple(keep_idxs)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(self.detections_per_image_train, topk_idxs.size(0))
                # torch.sort is actually faster than .topk (at least on GPUs)
                predicted_prob, idxs = predicted_prob.sort(descending=True)
                predicted_prob = predicted_prob[:num_topk]
                topk_idxs = topk_idxs[idxs[:num_topk]]

                anchor_idxs = topk_idxs // self.num_classes
                classes_idxs = topk_idxs % self.num_classes

                box_reg_i = box_reg_i[anchor_idxs]
                anchors_i = anchors_i[anchor_idxs]
                # predict boxes
                predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)

                boxes_all.append(predicted_boxes)
                scores_all.append(predicted_prob)
                class_idxs_all.append(classes_idxs)

            boxes_all, scores_all, class_idxs_all = [
                cat(x) for x in [boxes_all, scores_all, class_idxs_all]
            ]
            keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_thresh_train)
            keep = keep[: self.detections_per_image_train] # instead of self.max_detections_per_image

            results_per_image = Instances(image_size)
            results_per_image.pred_boxes = Boxes(boxes_all[keep])
            results_per_image.scores = scores_all[keep]
            results_per_image.pred_classes = class_idxs_all[keep]
            # end self.inference_single_image
            results.append(results_per_image)
        # end self.inference
        predictions = []
        for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes
        ):
            # height = input_per_image.get("height", image_size[0])
            # width = input_per_image.get("width", image_size[1])
            # r = detector_postprocess(results_per_image, height, width)
            r = results_per_image
            predictions.append({"instances": r})
        return predictions, loss_inputs

    def compute_losses(self, gt_instances, loss_inputs):
        gt_instances = [x.to(self.device) for x in gt_instances]
        anchors = loss_inputs['anchors']
        pred_logits = loss_inputs['pred_logits']
        pred_anchor_deltas = loss_inputs['pred_anchor_deltas']

        gt_labels, gt_boxes = self.label_anchors(anchors, gt_instances)
        losses = self.losses(anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes)
        return losses
