import torch
from typing import Dict, List
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from .adet import OneStageDetector


@META_ARCH_REGISTRY.register()
class CustomFCOS(OneStageDetector):
    '''
    With train_forward() and compute_losses(), we can separate
    the forward pass of inputs from the loss generation.
    '''
    def train_forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        '''
        Do not use the gt_instances of the inputs here.
        '''
        # if self.training:
        #     return super().forward(batched_inputs)
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)
        # start self.proposal_generator.forward: proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposal_features = [features[f] for f in self.proposal_generator.in_features]
        locations = self.proposal_generator.compute_locations(proposal_features)
        logits_pred, reg_pred, ctrness_pred, top_feats, bbox_towers = self.proposal_generator.fcos_head(
            proposal_features, None, self.proposal_generator.yield_proposal or self.proposal_generator.yield_box_feats
        )
        loss_inputs = {'logits_pred':logits_pred,
                                                  'reg_pred':reg_pred,
                                                  'ctrness_pred':ctrness_pred,
                                                  'locations':locations, 
                                                  'top_feats':top_feats
                                                  }
        #temporarily change the nms_thresh value of fcos_outputs
        self.nms_thresh_test = self.proposal_generator.fcos_outputs.nms_thresh
        self.proposal_generator.fcos_outputs.nms_thresh = self.nms_thresh_train
        proposals = self.proposal_generator.fcos_outputs.predict_proposals(
                logits_pred, reg_pred, ctrness_pred,
                locations, images.image_sizes, top_feats)
        self.proposal_generator.fcos_outputs.nms_thresh = self.nms_thresh_test
        # end self.proposal_generator.forward: proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        predictions = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            # height = input_per_image.get("height", image_size[0])
            # width = input_per_image.get("width", image_size[1])
            # r = detector_postprocess(results_per_image, height, width)
            r = results_per_image
            predictions.append({"instances": r})
        return predictions, loss_inputs

    def compute_losses(self, gt_instances, loss_inputs):
        gt_instances = [x.to(self.device) for x in gt_instances]
        logits_pred = loss_inputs['logits_pred']
        reg_pred = loss_inputs['reg_pred']
        ctrness_pred = loss_inputs['ctrness_pred']
        locations = loss_inputs['locations']
        top_feats = loss_inputs['top_feats']


        _, proposal_losses = self.proposal_generator.fcos_outputs.losses(
                logits_pred, reg_pred, ctrness_pred,
                locations, gt_instances, top_feats
            )
        return proposal_losses