import os
import torch
import argparse

from detectron2.config import CfgNode
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

from src.evaluation.build import build_evaluator
from src.models.build import build_model
from src.datasets.build import build_dataloaders

def test(run_path=None, cfg=None, model=None, val=False, test=True):
    if run_path is not None:
        assert cfg is None
        cfg = CfgNode(init_dict=CfgNode.load_yaml_with_base(filename=os.path.join(run_path, 'config.yaml')))
        cfg.OUTPUT_DIR = run_path
    else:
        assert cfg is not None

    logger = setup_logger(os.path.join(cfg.OUTPUT_DIR, 'evaluation'))
    for h in logger.handlers[:-2]: # remove handlers such that multiple runs can be tested in one script
        logger.removeHandler(h)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    val_evaluator = build_evaluator(cfg, val=True)
    test_evaluator = build_evaluator(cfg, val=False)

    if model is None:
        model_path = os.path.join(cfg.OUTPUT_DIR, 'final_model.pt')
        model = build_model(cfg)
        model.load_state_dict(torch.load(model_path))
    if isinstance(model, str):
        model_path = model
        model = build_model(cfg)
        model.load_state_dict(torch.load(model_path))
    model.to(cfg.MODEL.DEVICE)

    with EventStorage(0) as storage:
        if val:
            val_results = inference_on_dataset(model, val_loader, val_evaluator)
            val_score = val_results['score'] if 'score' in val_results else -val_results['bbox']['AP50']
            logger.info(val_results)
            logger.info('Val score: {}'.format(val_score))

        if test:
            test_results = inference_on_dataset(model, test_loader, test_evaluator)
            test_score = test_results['score'] if 'score' in test_results else -test_results['bbox']['AP50']
            logger.info(test_results)
            logger.info('Test score: {}'.format(test_score))
            
    return test_score if test else val_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation args.')
    parser.add_argument('--run_path', type=str, required=True, help='Path to the run you want to evaluate.')
    parser.add_argument('--val', type=bool, default=False, help='Whether to do validation.')
    parser.add_argument('--test', type=bool, default=True, help='Whether to do testing.')

    args = parser.parse_args()
    score = test(run_path=args.run_path, val=args.val, test=args.test)