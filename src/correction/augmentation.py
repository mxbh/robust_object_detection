import numpy as np
import torch
from copy import deepcopy
from detectron2.config.config import configurable
from detectron2.data.transforms.augmentation import AugInput, Augmentation
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform, NoOpTransform
from scipy.ndimage.filters import gaussian_filter



class BatchAugmentation:
    '''
    Wrapper around detectrons Augmentation to allow for augmentations on batches with tensors.
    '''
    @configurable
    def __init__(self, augmentation):
        self.augmentation = augmentation
        self.transforms = []

    @classmethod
    def from_config(cls, cfg):
        aug_list = []
        for aug_string in cfg.CORRECTION.AUG:
            aug_string_splitted = aug_string.split('-')
            aug_type = aug_string_splitted[0]

            if aug_type == 'hflip':
                p = float(aug_string_splitted[1])
                aug_list.append(T.RandomFlip(prob=p, horizontal=True, vertical=False))
            elif aug_type == 'vflip':
                p = float(aug_string_splitted[1])
                aug_list.append(T.RandomFlip(prob=p, horizontal=False, vertical=True))
            elif aug_type == 'brightness':
                intensity_min, intensity_max = float(aug_string_splitted[1]), float(aug_string_splitted[2])
                aug_list.append(T.RandomBrightness(intensity_min, intensity_max))
            elif aug_type == 'contrast':
                intensity_min, intensity_max = float(aug_string_splitted[1]), float(aug_string_splitted[2])
                aug_list.append(T.RandomContrast(intensity_min, intensity_max))
            elif aug_type == 'saturation':
                intensity_min, intensity_max = float(aug_string_splitted[1]), float(aug_string_splitted[2])
                aug_list.append(T.RandomSaturation(intensity_min, intensity_max))
            elif aug_type == 'erase':
                p, max_size = float(aug_string_splitted[1]), int(aug_string_splitted[2])
                aug_list.append(RandomErase(p, max_size))
            elif aug_type == 'blur':
                p, sig = float(aug_string_splitted[1]), float(aug_string_splitted[2])
                aug_list.append(RandomBlur(p, sig))
        
        return {'augmentation': T.AugmentationList(aug_list)}

    def apply_batch(self, batch):
        self.transforms = []

        new_batch = []
        for sample in batch:
            new_sample = {key:value for key,value in sample.items()}
            new_sample['instances'] = deepcopy(new_sample['instances'])

            image = numpy_image(sample['image'])
            boxes = sample['instances'].gt_boxes.tensor.numpy()
            input_ = AugInput(image=image, boxes=boxes)
            
            tfm = self.augmentation(input_)
            self.transforms.append(tfm)

            new_sample['image'] = torch_image(input_.image)
            new_sample['instances'].gt_boxes.tensor = torch.from_numpy(input_.boxes)

            new_batch.append(new_sample)

        return new_batch
    
    def apply_gt_instances(self, instance_list, inverse=False, inplace=False):
        if not inplace:
            instance_list = deepcopy(instance_list)
        if inverse:
            assert self.transforms != []
                
        for instances, tfm in zip(instance_list, self.transforms):
            if inverse:
                tfm = tfm.inverse()

            boxes = instances.gt_boxes.tensor.detach().cpu().numpy()
            boxes_transformed = tfm.apply_box(boxes)
            instances.gt_boxes.tensor = torch.from_numpy(boxes_transformed)

        return instance_list

def numpy_image(img):
    return img.detach().cpu().permute(1,2,0).numpy()

def torch_image(img):
    return torch.from_numpy(np.copy(img)).permute(2,0,1)

class RandomErase(Augmentation):
    def __init__(self, prob=0.5, max_size=64):
        super().__init__()
        self.prob = prob
        self.max_size = max_size

    def get_transform(self, image):
        H, W = image.shape[:2]
        if np.random.uniform() < self.prob:
            x = np.random.randint(low=0, high=W)
            y = np.random.randint(low=0, high=H)
            h,w = np.random.randint(low=self.max_size // 2, high=self.max_size, size=2)

            x1 = max(0, x - w // 2)
            x2 = min(W, x + w // 2)
            y1 = max(0, y - h // 2)
            y2 = min(H, y + h // 2)

            return BoxEraseTransform(x1=x1,x2=x2,y1=y1,y2=y2)
        else:
            return NoOpTransform()

class BoxEraseTransform(Transform):
    def __init__(self, x1, x2, y1, y2):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        tensor[self.y1:self.y2,self.x1:self.x2] = 0
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is a no-op.
        """
        return NoOpTransform()


class RandomBlur(Augmentation):
    def __init__(self, prob, sigma):
        super().__init__()
        self.prob = prob
        self.sigma = sigma

    def get_transform(self, image):
        if np.random.uniform() < self.prob:
            return BlurTransform(sigma=self.sigma)
        else:
            return NoOpTransform()

class BlurTransform(Transform):
    def __init__(self, sigma):
            super().__init__()
            self.sigma = sigma

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return gaussian_filter(img, sigma=(self.sigma, self.sigma, 0))

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is a no-op.
        """
        return NoOpTransform()

