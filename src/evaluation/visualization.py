import numpy as np
from PIL import Image
from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.visualizer import Visualizer as D2Visualizer, _create_text_labels


class Visualizer:
    def __init__(self, dataset=None, rgb=False, scale=1.0, labels=True, color=None):
        self.dataset = dataset
        self.metadata = MetadataCatalog.get(dataset)
        self.class_names = self.metadata.thing_classes 
        self.rgb = rgb
        self.scale = scale
        self.labels = labels
        self.color = color

    def visualize_batch_ground_truth(self, dict_):
        img = dict_['image'].permute(1,2,0).numpy()
        img = img[:,:,::-1] if not self.rgb else img
        vis = D2Visualizer(img_rgb=img, metadata=self.metadata, scale=self.scale)
        vis._default_font_size = 12. / self.scale
    
        objects = dict_['ground_truth']
        boxes = np.stack([np.array(obj['bbox']) for obj in objects])
        # scale boxes
        orig_height, orig_width = dict_['height'], dict_['width']
        actual_height, actual_width = img.shape[:2]
        boxes[:,[0,2]] *= actual_width / orig_width
        boxes[:,[1,3]] *= actual_height / orig_height

        class_ids = [obj['category_id'] for obj in objects]
        labels = _create_text_labels(classes=class_ids, scores=None, class_names=self.class_names) \
                    if self.labels else None

        colors = None if self.color is None else ([self.color] * len(boxes))
        result = vis.overlay_instances(boxes=boxes, labels=labels, assigned_colors=colors)

        return Image.fromarray(result.get_image())



    def visualize_batch_annotation(self, dict_, instances=None):
        img = dict_['image'].permute(1,2,0).numpy()
        img = img[:,:,::-1] if not self.rgb else img
        vis = D2Visualizer(img_rgb=img, metadata=self.metadata, scale=self.scale)
        vis._default_font_size = 12. / self.scale
        
        if instances is None:
            instances = dict_['instances']
        boxes = instances.gt_boxes.tensor.numpy()
        class_ids = instances.gt_classes.numpy()
        labels = _create_text_labels(classes=class_ids, scores=None, class_names=self.class_names) \
                    if self.labels else None
        colors = None if self.color is None else ([self.color] * len(boxes))
        result = vis.overlay_instances(boxes=boxes, labels=labels, assigned_colors=colors)

        return Image.fromarray(result.get_image())  



    def visualize_model_output(self, input_dict, output_dict, threshold=0.5, rescale=True):
        img = input_dict['image'].permute(1,2,0).numpy()
        img = img[:,:,::-1] if not self.rgb else img
        vis = D2Visualizer(img_rgb=img, metadata=self.metadata, scale=self.scale)
        vis._default_font_size = 12. / self.scale

        boxes = output_dict['instances'].pred_boxes.tensor.detach().cpu().numpy()
        class_ids = output_dict['instances'].pred_classes.detach().cpu().numpy()
        scores = output_dict['instances'].scores.detach().cpu().numpy()
        keep = scores >= threshold

        if rescale: # scale boxes
            orig_height, orig_width = input_dict['height'], input_dict['width']
            actual_height, actual_width = img.shape[:2]
            boxes[:,[0,2]] *= actual_width / orig_width
            boxes[:,[1,3]] *= actual_height / orig_height

        labels = _create_text_labels(classes=class_ids[keep], scores=scores[keep], class_names=self.class_names) \
                    if self.labels else None
        colors = None if self.color is None else ([self.color] * len(boxes))
        result = vis.overlay_instances(boxes=boxes[keep], labels=labels, assigned_colors=colors)

        return Image.fromarray(result.get_image()) 