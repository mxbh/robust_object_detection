import os
import numpy as np
import torch
from copy import deepcopy
from typing import OrderedDict

from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.events import CommonMetricPrinter, EventStorage
from detectron2.engine import default_writers
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm

from src.evaluation.build import build_evaluator
from src.models.build import build_model
from src.datasets.build import build_dataloaders
from src.config.utils import set_output_dir
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.correction.correction_module import CorrectionModule
from src.correction.augmentation import BatchAugmentation


def train(cfg, model=None):
    set_output_dir(cfg)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter=cfg.SOLVER.MAX_ITER)
    logger = setup_logger(cfg.OUTPUT_DIR)
    for h in logger.handlers[:-2]: # remove handlers such that multiple runs don't affect each other
        logger.removeHandler(h)
    logger.info('Output dir: {}'.format(cfg.OUTPUT_DIR))

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    evaluator = build_evaluator(cfg)

    if model is None:
        student = build_model(cfg)
    else: 
        student = model
    teacher = None

    optimizer = build_optimizer(cfg, student)
    scheduler = build_lr_scheduler(cfg, optimizer)
    correction_module = CorrectionModule(cfg)
    augmentation = BatchAugmentation(cfg)

    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER
    best_score = np.inf
    patience_left = cfg.SOLVER.PATIENCE

    if cfg.SOLVER.USE_CHECKPOINT is not None and cfg.SOLVER.USE_CHECKPOINT != '':
        model_state_dict,\
            optimizer_state_dict,\
            scheduler_state_dict,\
            checkpoint_iter,\
            teacher_model_state_dict = load_checkpoint(cfg.SOLVER.USE_CHECKPOINT)
        student.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        scheduler._max_iter = max_iter + 1
        start_iter = checkpoint_iter
        teacher = deepcopy(student)
        teacher.load_state_dict(teacher_model_state_dict)

    student.to(cfg.MODEL.DEVICE)
    keep_rate = cfg.MODEL.EMA_TEACHER.KEEP_RATE

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(train_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            b = len(data)
            data_student = augmentation.apply_batch(data)
            predictions_student, loss_inputs_student = student.train_forward(data_student)

            if iteration > cfg.CORRECTION.WARMUP:
                if teacher is None:
                    # init EMA teacher
                    teacher = deepcopy(student)
                    
                data_teacher = data
                target_instances_teacher = [x['instances'] for x in data_teacher]
                predictions_teacher, loss_inputs_teacher = teacher.train_forward(data_teacher)

                target_instances_student = correction_module(
                    [x['instances'] for x in predictions_teacher], 
                    [x.to(student.device) for x in target_instances_teacher]
                )
                target_instances_student = augmentation.apply_gt_instances(target_instances_student, inverse=False)
            else: 
                target_instances_student = [x['instances'] for x in data_student]

            loss_dict = student.compute_losses(gt_instances=target_instances_student,
                                               loss_inputs=loss_inputs_student)

            losses = sum(loss_dict.values())
            loss_dict_reduced = {k: v.item()
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            storage.put_scalars(train_loss=losses_reduced, **loss_dict_reduced)

            if not torch.isfinite(losses).all():
                for writer in writers:
                    if isinstance(writer, CommonMetricPrinter):
                        logger.info('NaN in losses: {}'.format(loss_dict))
                    writer.write()
                assert False

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                'lr', optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration > cfg.CORRECTION.WARMUP:
                # update EMA teacher
                with torch.no_grad():
                    student_model_dict = student.state_dict()

                    new_teacher_dict = OrderedDict()
                    for key, value in teacher.state_dict().items():
                        if key in student_model_dict.keys():
                            new_teacher_dict[key] = student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                        else:
                            raise Exception("{} is not found in student model".format(key))

                    teacher.load_state_dict(new_teacher_dict)

            do_test = False
            if (cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0) or (iteration == max_iter - 1):
                do_test = True
                
                model = teacher if teacher is not None else student
                results = inference_on_dataset(model, val_loader, evaluator)

                val_score = results['score'] if 'score' in results else - \
                    results['bbox']['AP50']
                storage.put_scalar('val_score', val_score,
                                   smoothing_hint=False)

                if val_score < best_score:
                    best_score = val_score
                    model.cpu()
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'final_model.pt'))
                    model.to(cfg.MODEL.DEVICE)
                    patience_left = cfg.SOLVER.PATIENCE
                elif val_score == best_score:
                    patience_left = cfg.SOLVER.PATIENCE
                else:
                    patience_left -= 1

            if ((iteration + 1) % 20 == 0) or (iteration == max_iter - 1) or (do_test):
                for writer in writers:
                    writer.write()
                    if do_test and isinstance(writer, CommonMetricPrinter):
                        logger.info('val_score: {}, best_val_score: {}'.format(val_score, best_score))

            if (iteration + 1) in cfg.SOLVER.CHECKPOINTS:
                save_checkpoint(student, optimizer, scheduler, iteration, cfg.OUTPUT_DIR, teacher_model=teacher)
                logger.info('Saved checkpoint {}'.format(iteration))

            if patience_left < 0:
                logger.info('Early stopping!')
                break
    try:    
        model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'final_model.pt')))
    except:
        pass
    logger.info('DONE.')
    return model, best_score
