from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper, transforms as T
from .register import register_datasets

def build_dataloaders(cfg):
    register_datasets(cfg)
    
    augs = build_train_augmentation(cfg)
    mapper = DatasetMapper(
        is_train=True,
        augmentations=augs,
        image_format=cfg.INPUT.FORMAT,
        use_instance_mask=cfg.MODEL.MASK_ON,
        use_keypoint=cfg.MODEL.KEYPOINT_ON,
        instance_mask_format=cfg.INPUT.MASK_FORMAT,
        keypoint_hflip_indices=None,
        precomputed_proposal_topk=None,
        recompute_boxes=False,
    )
    train_loader = build_detection_train_loader(cfg, mapper=mapper)#, cfg.DATASETS.TRAIN)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.VAL)
    val_loader.dataset._map_func._obj.is_train = True # ensures that annotations are not discarded 
    test_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
    test_loader.dataset._map_func._obj.is_train = True # ensures that annotations are not discarded 

    return train_loader, val_loader, test_loader


def build_train_augmentation(cfg):
    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    
    if cfg.INPUT.RANDOM_FLIP == "horizontal":
        augmentation.append(
            T.RandomFlip(
                horizontal=True,
                vertical=False,
            )
        )
    elif cfg.INPUT.RANDOM_FLIP == "vertical":
        augmentation.append(
            T.RandomFlip(
                horizontal=False,
                vertical=True,
            )
        )
    elif cfg.INPUT.RANDOM_FLIP in ["vertical&horizontal",
                                   "horizontal&vertical",
                                   "both"]:
        augmentation.append(
            T.RandomFlip(
                horizontal=False,
                vertical=True,
            )
        )
        augmentation.append(
            T.RandomFlip(
                horizontal=True,
                vertical=False,
            )
        )
    if cfg.INPUT.CROP.ENABLED:
        augmentation.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    return augmentation