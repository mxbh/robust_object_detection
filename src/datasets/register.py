from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.pascal_voc import load_voc_instances, CLASS_NAMES as VOC_CLASS_NAMES
from .nwpu import *
from .noise import load_noisy_instances, build_label_transform


def register_datasets(cfg):
    train_datasets = cfg.DATASETS.TRAIN
    val_datasets = cfg.DATASETS.VAL
    test_datasets = cfg.DATASETS.TEST
    if isinstance(train_datasets, str):
        train_datasets = [train_datasets]
    if isinstance(val_datasets, str):
        val_datasets = [val_datasets]
    if isinstance(test_datasets, str):
        test_datasets = [test_datasets]

    if 'voc_2007_custom' in ''.join(train_datasets) or 'voc_2007_custom' in ''.join(val_datasets) or 'voc_2007_custom' in ''.join(test_datasets):
        for split in ['custom_train', 'custom_val']:
            name = 'voc_2007_' + split
            try:
                DatasetCatalog.remove(name)
            except:
                pass

            year = '2007'
            dirname = os.path.join(
                os.getenv("DETECTRON2_DATASETS", "datasets"), 'VOC{}'.format(year))
            DatasetCatalog.register(name, lambda split=split, dirname=dirname: load_voc_instances(dirname=dirname,
                                                                                 split=split,
                                                                                 class_names=list(VOC_CLASS_NAMES)))
            MetadataCatalog.get(name).set(thing_classes=list(VOC_CLASS_NAMES),
                                          evaluator_type='pascal_voc',
                                          dirname=dirname,
                                          year=int(year),
                                          split=split)

    if 'nwpu' in ''.join(train_datasets) or 'nwpu' in ''.join(val_datasets) or 'nwpu' in ''.join(test_datasets):
        for split in ['train', 'val', 'test']:
            name = 'nwpu_' + split
            try:
                DatasetCatalog.remove(name)
            except:
                pass
            DatasetCatalog.register(
                name, lambda split=split: load_nwpu_instances(split=split))
            MetadataCatalog.get(
                name).set(thing_classes=NWPU_CLASS_NAMES, evaluator_type='coco')
     
    if 'noisy' in ''.join(train_datasets) or 'noisy' in ''.join(val_datasets) or 'noisy' in ''.join(test_datasets):
        transform = build_label_transform(cfg)
        if 'nwpu' in ''.join(train_datasets) or 'nwpu' in ''.join(val_datasets) or 'nwpu' in ''.join(test_datasets):
            for split in ['train', 'val', 'test']:
                name = 'noisy_nwpu_' + split
                try:
                    DatasetCatalog.remove(name)
                except:
                    pass
                DatasetCatalog.register(name,
                                        lambda split=split: load_noisy_instances(dicts=load_nwpu_instances(split=split),
                                                                                 transform=transform))
                MetadataCatalog.get(name).set(thing_classes=NWPU_CLASS_NAMES,
                                              evaluator_type='coco')

        if 'voc' in ''.join(train_datasets) or 'voc' in ''.join(val_datasets) or 'voc' in ''.join(test_datasets):
            for year in ['2007', '2012']:

                if year == '2007':
                    splits = ['train', 'val', 'test', 'trainval', 'custom_train', 'custom_val']
                elif year == '2012':
                    splits = ['train', 'val', 'test', 'trainval']

                for split in splits:
                    dirname = os.path.join(
                        os.getenv("DETECTRON2_DATASETS", "datasets"), 'VOC{}'.format(year))
                    name = 'noisy_voc_{}_{}'.format(year, split)
                    try:
                        DatasetCatalog.remove(name)
                    except:
                        pass
                    DatasetCatalog.register(name,
                                            lambda split=split, year=year: load_noisy_instances(dicts=DatasetCatalog.get('voc_{}_{}'.format(year, split)),
                                                                                     transform=transform))
                    MetadataCatalog.get(name).set(thing_classes=list(VOC_CLASS_NAMES),
                                                  evaluator_type='pascal_voc',
                                                  dirname=dirname,
                                                  year=int(year),
                                                  split=split)
