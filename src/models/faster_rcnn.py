import torch
from typing import Dict, List
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage


@META_ARCH_REGISTRY.register()
class CustomFasterRCNN(GeneralizedRCNN):
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

        # start proposal_generator.forward: proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposal_generator_features = [features[f] for f in self.proposal_generator.in_features]
        anchors = self.proposal_generator.anchor_generator(proposal_generator_features)
        pred_objectness_logits, pred_anchor_deltas = self.proposal_generator.rpn_head(proposal_generator_features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits 
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.proposal_generator.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]
        #gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
        #losses = self.losses(
        #        anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
        #)
        loss_inputs['proposal_generator'] = dict(anchors=anchors,
                                                      pred_objectness_logits=pred_objectness_logits,
                                                      pred_anchor_deltas=pred_anchor_deltas)

        proposals = self.proposal_generator.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
        )
        # end proposal_generator.forward

        # start roi_heads.forward: _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # start roi_heads._forward_box: pred_instances = self._forward_box(features, proposals)
        roi_heads_features = [features[f] for f in self.roi_heads.box_in_features]
        loss_inputs['roi_heads'] = dict(features=roi_heads_features,
                                             proposals=proposals)
        roi_heads_box_features = self.roi_heads.box_pooler(roi_heads_features, [x.proposal_boxes for x in proposals])
        roi_heads_box_features = self.roi_heads.box_head(roi_heads_box_features)
        roi_heads_predictions = self.roi_heads.box_predictor(roi_heads_box_features)
        # start roi_heads.box_predictor.inference: pred_instances, _ = self.roi_heads.box_predictor.inference(roi_heads_predictions, proposals)
        boxes = self.roi_heads.box_predictor.predict_boxes(roi_heads_predictions, proposals)
        scores = self.roi_heads.box_predictor.predict_probs(roi_heads_predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        pred_instances, _ = fast_rcnn_inference(boxes=boxes,
                                             scores=scores,
                                             image_shapes=image_shapes,
                                             score_thresh=self.score_thresh_train,
                                             nms_thresh=self.nms_thresh_train,
                                             topk_per_image=self.detections_per_image_train, 
                                             # initialized in build_model()
                                            ) 
        # end roi_heads.box_predictor.inference
        # end roi_heads._forward_box
        # end roi_heads.forward

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        # postprocessing to obtain proper predictions
        assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        #predictions = GeneralizedRCNN._postprocess(pred_instances, batched_inputs, images.image_sizes) # this brings data to original scale (before padding AND scaling)
        predictions = [{'instances': x} for x in pred_instances]
        return predictions, loss_inputs


    def compute_losses(self, gt_instances, loss_inputs):
        gt_instances = [x.to(self.device) for x in gt_instances]

        # proposal losses
        anchors = loss_inputs['proposal_generator']['anchors']
        pred_objectness_logits = loss_inputs['proposal_generator']['pred_objectness_logits']
        pred_anchor_deltas = loss_inputs['proposal_generator']['pred_anchor_deltas']
        
        gt_labels, gt_boxes = self.proposal_generator.label_and_sample_anchors(anchors, gt_instances)
        proposal_losses = self.proposal_generator.losses(
            anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
        )

        # roi_heads/detector losses
        targets = gt_instances
        proposals = loss_inputs['roi_heads']['proposals']
        features = loss_inputs['roi_heads']['features']

        proposals = self.roi_heads.label_and_sample_proposals(proposals, targets)
        box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)
        detector_losses = self.roi_heads.box_predictor.losses(predictions, proposals)

        ##########################################################
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses