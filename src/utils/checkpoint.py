import os
import torch

def save_checkpoint(model, optim, scheduler, iteration, output_dir, teacher_model=None):
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = dict(iteration=iteration,
                      model=model.state_dict(),
                      optim=optim.state_dict(),
                      scheduler=scheduler.state_dict())
    if teacher_model is not None:
        checkpoint['teacher_model'] = teacher_model.state_dict()
    torch.save(checkpoint, os.path.join(checkpoint_dir, str(iteration) + '.pt'))

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint['model']
    optim_state_dict = checkpoint['optim']
    scheduler_state_dict = checkpoint['scheduler']
    iteration = checkpoint['iteration']

    if 'teacher_model' in checkpoint.keys():
        teacher_model_state_dict = checkpoint['teacher_model']
        return model_state_dict, optim_state_dict, scheduler_state_dict, iteration, teacher_model_state_dict
    else:
        return model_state_dict, optim_state_dict, scheduler_state_dict, iteration
    
