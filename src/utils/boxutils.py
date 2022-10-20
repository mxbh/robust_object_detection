from torchvision.ops import box_iou
import torch
import numpy as np

def box_iou_distance(boxes1, boxes2):
    '''
    Returned values are 1 - IoU, i.e. values in [0,1].
    '''
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1)
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2)
    return 1 - box_iou(boxes1, boxes2) # (n1,n2)

def box_center_distance(boxes1, boxes2):
    '''
    Returned values are the euclidean center distances normalized by the sizes of the boxes.
    '''
    if isinstance(boxes1, np.ndarray):
        boxes1 = torch.from_numpy(boxes1)
    if isinstance(boxes2, np.ndarray):
        boxes2 = torch.from_numpy(boxes2)

    centers1 = 0.5 * torch.stack([boxes1[:,0] + boxes1[:,2],
                                  boxes1[:,1] + boxes1[:,3]], dim=1)
    centers2 = 0.5 * torch.stack([boxes2[:,0] + boxes2[:,2],
                                  boxes2[:,1] + boxes2[:,3]], dim=1)
    sizes1 = 0.5 * (boxes1[:,2] - boxes1[:,0] + boxes1[:,3] - boxes1[:,1])
    sizes2 = 0.5 * (boxes2[:,2] - boxes2[:,0] + boxes2[:,3] - boxes2[:,1])

    distances = torch.cdist(centers1, centers2, p=2)
    distances = distances / torch.sqrt(sizes1).reshape(-1, 1)
    distances = distances / torch.sqrt(sizes2).reshape( 1,-1)
    return distances                    
