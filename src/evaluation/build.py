import os
import json
from detectron2.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from detectron2.data.datasets.coco import convert_to_coco_dict
from detectron2.data import MetadataCatalog


def build_evaluator(cfg, val=True):
    if val:
        dataset_name = cfg.DATASETS.VAL
    else:
        dataset_name = cfg.DATASETS.TEST
        
    if not isinstance(dataset_name, str):
        assert len(dataset_name) == 1
        dataset_name = dataset_name[0]
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type == 'pascal_voc':
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == 'coco':
        if not hasattr(MetadataCatalog.get(dataset_name), "json_file"):
            # create and register coco json first
            # we do this manually because convert_to_coco_json() gets stuck with file_lock()
            coco_dict = convert_to_coco_dict(dataset_name)
            json_path = os.path.join(cfg.OUTPUT_DIR, dataset_name + '_coco_format.json')
            with open(json_path, 'w') as f:
                    json.dump(coco_dict, f)
            MetadataCatalog.get(dataset_name).set(json_file=json_path)
        
        return COCOEvaluator(dataset_name=dataset_name,
                             tasks=('bbox',),
                             distributed=False,
                             output_dir=cfg.OUTPUT_DIR)
    else:
        raise NotImplementedError('Unknow evaluator type {}!'.format(evaluator_type))
