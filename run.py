import argparse
from detectron2.config import CfgNode
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True, type=str, help='Which training method to use (standard vs robust).')
parser.add_argument('--runs', required=False, default=1, type=int, help='How many training runs.')
parser.add_argument('--test', required=False, default=False, type=bool, help='Whether to test.')
parser.add_argument('--config', required=True, type=str, help='Config file to use')
args = parser.parse_args()

with open(args.config, 'r') as f: 
    cfg = CfgNode.load_cfg(f)

if args.method == 'standard':
    from src.engine.train_standard import train
elif args.method == 'robust':
    from src.engine.train_teacher_student import train
else:
    assert False, 'Unknown train method.'

for run in range(args.runs):
    cfg_run = deepcopy(cfg)
    model, best_score = train(cfg_run)
    if args.test:
        from test import test
        test_score = test(run_path=None, cfg=cfg_run, model=model, val=False, test=True)