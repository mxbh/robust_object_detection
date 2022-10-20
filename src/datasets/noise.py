import torch
from copy import deepcopy
from detectron2.config.config import CfgNode
from ..utils.misc import get_random_generator


def load_noisy_instances(dicts, transform=lambda x:x['annotations']):
    '''
    Adds noisy instances to the instances in dicts usingthe provided transform.
    '''
    for d in dicts:
        d['ground_truth'] = deepcopy(d['annotations'])
        d['annotations'] = transform(d)
    return dicts


def build_label_transform(cfg):
    num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
    assert num_classes == cfg.MODEL.ROI_HEADS.NUM_CLASSES

    try:
        noise_cfg = cfg.DATASETS.NOISE
    except:
        noise_cfg = CfgNode()
    
    transforms_list = []
    for key, value in noise_cfg.items():
        if key == 'UNIFORM_BBOX_NOISE':
            p = value.get('P')
            if p != 0:
                transforms_list.append(UniformBoxNoise(p=p))
        elif key == 'DROP_LABELS':
            p = value.get('P')
            if p != 0:
                transforms_list.append(DropLabels(p=p))
        else:
            raise ValueError(f'Unknown noise transform: {key}')

    if len(transforms_list) == 0:
        transforms_list.append(lambda x:x['annotations'])

    return CompositeLabelTransform(*transforms_list)

class CompositeLabelTransform:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            annotations = t(sample)
            sample['annotations'] = annotations
        return annotations

class UniformBoxNoise:
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        annotations = sample['annotations']
        boxes = torch.tensor([box['bbox'] for box in annotations]).reshape(-1,4)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        # fix seed with generator
        generator = get_random_generator(sample['image_id'])

        # sample noise
        noise = 2 * self.p * \
            torch.rand(boxes.shape, generator=generator) - self.p  # [-p,+p]
        noise[:, [0, 2]] = noise[:, [0, 2]] * w.view(-1, 1)  # [-wp,+wp]
        noise[:, [1, 3]] = noise[:, [1, 3]] * h.view(-1, 1)  # [-hp,+hp]
        boxes = boxes + noise

        # remove degenerate boxes
        mask = torch.logical_and(boxes[:, 2] > boxes[:, 0], boxes[:, 3] > boxes[:, 1])

        # reverse order because of popping
        for i in range(len(annotations))[::-1]:
            annotations[i]['bbox'] = boxes[i].tolist()
            if not mask[i]:
                annotations.pop(i)
        return annotations


class DropLabels:
    # drop
    def __init__(self, p):
        '''
        :param p: Ratio of annotations to delete. One means that one annotation per image is kept.
        '''
        self.p = p

    def __call__(self, sample):
        annotations = sample['annotations']
        num_objects = len(annotations)
        if num_objects > 0:
            # fix seed with generator
            generator = get_random_generator(sample['image_id'])
            if self.p == 1:
                mask = torch.full((num_objects,), False)
                i = torch.randint(low=0, high=num_objects, size=(1,))
                mask[i] = True
            else:
                mask = self.p < torch.rand(num_objects, generator=generator)


            # go through in reverse order
            for i in range(num_objects)[::-1]:
                if not mask[i]:
                    annotations.pop(i)
        return annotations
  