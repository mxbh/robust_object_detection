import torch
from detectron2.modeling import build_model as build_architecture
from .faster_rcnn import CustomFasterRCNN
from .retinanet import CustomRetinaNet
from .fcos import CustomFCOS

def build_model(cfg):
    '''
    Wrapper around detectron's build_model() to do the backbone (or the whole model) initialization myself.
    '''
    # create new model
    model = build_architecture(cfg)
    if cfg.MODEL.WEIGHTS == 'detectron2://ImageNetPretrained/MSRA/R-50.pkl':
        model.backbone.bottom_up.load_state_dict(torch.load('./pretrained_backbones/ImageNetPretrained-MSRA-R-50.pt',
                                                    map_location=cfg.MODEL.DEVICE))
        print('Init model from MSRA.')
        assert cfg.INPUT.FORMAT == 'BGR', 'Input format for MSRA should be BGR!'

    elif cfg.MODEL.WEIGHTS == 'detectron2://ImageNetPretrained/torchvision/R-50.pkl':
        model.backbone.bottom_up.load_state_dict(torch.load('./pretrained_backbones/ImageNetPretrained-torchvision-R-50.pt',
                                                    map_location=cfg.MODEL.DEVICE))
        print('Init model from torchvision.')
        assert cfg.INPUT.FORMAT == 'RGB', 'Input format for torchvision should be RGB!'
        
    elif cfg.MODEL.WEIGHTS:
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS,
                                         map_location=cfg.MODEL.DEVICE))
        print(f'Init model from {cfg.MODEL.WEIGHTS}')
    else:
        print('Init model randomly.')

    # TODO: move this to constructors
    if isinstance(model, CustomFasterRCNN) or isinstance(model, CustomRetinaNet):
        model.nms_thresh_train = cfg.CORRECTION.get('NMS_THRESH_TRAIN', None)
        model.score_thresh_train = cfg.CORRECTION.get('SCORE_THRESH_TRAIN', None)
        model.detections_per_image_train = cfg.CORRECTION.get('DETECTIONS_PER_IMAGE_TRAIN', None)
    if isinstance(model, CustomFCOS):
        assert cfg.CORRECTION.DETECTIONS_PER_IMAGE_TRAIN == cfg.MODEL.FCOS.POST_NMS_TOPK_TRAIN
        assert cfg.CORRECTION.SCORE_THRESH_TRAIN == cfg.MODEL.FCOS.INFERENCE_TH_TRAIN
        model.nms_thresh_train = cfg.CORRECTION.get('NMS_THRESH_TRAIN', None)
        model.nms_thresh_train = cfg.MODEL.FCOS.NMS_TH 

    return model
