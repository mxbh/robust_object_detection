import os
from PIL import Image

NWPU_ROOT_DIR = './datasets/NWPU'
NWPU_CLASS_NAMES = ('airplane',
                    'ship',
                    'storage tank',
                    'baseball diamond',
                    'tennis court',
                    'basketball court',
                    'ground track field',
                    'harbor',
                    'bridge',
                    'vehicle')

def get_annotations(annotation_path):
    annotations = []

    with open(annotation_path, 'r') as f:  # encoding='utf-8-sig'
        content = f.read()
        objects = content.split('\n')
        objects = [x for x in objects if len(x) > 0]
        for obj in objects:
            info = obj.replace('(', '').replace(')', '').strip().split(',')
            assert len(info) == 5, 'wronging occurred in label convertion!!'
            label = int(info[4]) - 1  # -1 because we want to start with 0
            x1, y1, x2, y2 = [float(x) for x in info[:4]]
            
            annotations.append(dict(bbox=[x1,y1,x2,y2],
                                    bbox_mode=0, #BoxMode.XYXY_ABS
                                    category_id=label))  
    return annotations

def load_nwpu_instances(split, label_transform=None):
    with open(os.path.join(NWPU_ROOT_DIR, 'Splits/{}_set_positive.txt'.format(split)), 'r') as f:
        positive_imgs = f.readlines()
    with open(os.path.join(NWPU_ROOT_DIR, 'Splits/{}_set_negative.txt'.format(split)), 'r') as f:
        negative_imgs = f.readlines()

    dicts = []

    for name in positive_imgs:
        name = name.strip()
        img_path = os.path.join(NWPU_ROOT_DIR, 'positive image set', name)
        annotation_path = os.path.join(NWPU_ROOT_DIR, 'ground truth',  name[:-3]+'txt')

        img = Image.open(img_path)
        width, height = img.size
        img_id = os.path.join('positive image set', name)
        annotations = get_annotations(annotation_path)

        dicts.append(dict(file_name=img_path,
                          height=height,
                          width=width,
                          image_id=img_id,
                          annotations=annotations
                          ))

    for name in negative_imgs:
        name = name.strip()
        img_path = os.path.join(NWPU_ROOT_DIR, 'negative image set', name)
        img = Image.open(img_path)
        width, height = img.size
        img_id = os.path.join('negative image set', name)

        dicts.append(dict(file_name=img_path,
                          height=height,
                          width=width,
                          image_id=img_id,
                          annotations=[]
                          ))

    return dicts

    


