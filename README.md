# Robust Object Detection

This repository contains the source code for the paper "Robust Object Detection in Remote Sensing Imagery with Noisy
and Sparse Geo-Annotations".

> Recently, the availability of remote sensing imagery from aerial vehicles and satellites constantly improved. For an automated interpretation of such data, deep-learning-based object detectors achieve state-of-the-art performance. However, established object detectors require complete, precise, and correct bounding box annotations for training. In order to create the necessary training annotations for object detectors, imagery can be georeferenced and combined with data from other sources, such as points of interest localized by GPS sensors. Unfortunately, this combination often leads to poor object localization and missing annotations. Therefore, training object detectors with such data often results in insufficient detection performance. In this paper, we present a novel approach for training object detectors with extremely noisy and incomplete annotations. Our method is based on a teacher-student learning framework and a correction module accounting for imprecise and missing annotations. Thus, our method is easy to use and can be combined with arbitrary object detectors. We demonstrate that our approach improves standard detectors by 37.1% AP_50 on a noisy real-world remote-sensing dataset. Furthermore, our method achieves great performance gains on two datasets with synthetic noise. Code is available at https://github.com/mxbh/robust_object_detection.

An extended version of the paper with more detailed explanations can be found [here](http://arxiv.org/abs/2210.12989).

## Usage 
1. Download the desired datasets and place them into `./datasets`.
2. For NWPU and Pascal VOC2007, place the provided split files under the corresponding dataset folder.
```
datasets/
    NWPU/
        ground truth/
            ...
        negative image set/
            ...
        positive image set/
            ...
        Splits/
            test_set_negative.txt
            test_set_positive.txt
            train_set_negative.txt
            train_set_positive.txt
            val_set_negative.txt
            val_set_positive.txt
    VOC2007/
        ImageSets/
            Main/
                custom_train.txt
                custom_val.txt
                ...
            ...
        ...
    VOC2012/
        ...

```
3. Install the requirements, particulary we used `torch==1.9.1`, `torchvision==0.10.1` and `detectron2==0.5`.
A superset of the required packages is listed in `./requirements.txt` and `environment.yaml`.
4. Download [pretrained backbone weights](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) (other backbones can be found [here](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#imagenet-pretrained-models)) into `./pretrained_backbones`. Make sure that the files are named the same way in `./src/models/build.py`.
5. Optional: Modify the config files provided in `./configs`.
6. Run the training scripts, e.g.
```
python run.py \
    --method=standard \
    --config=./configs/voc/faster_rcnn_standard_Nb=40_Ns=0.yaml

python run.py \
    --method=robust \
    --config=./configs/voc/faster_rcnn_robust_Nb=40_Ns=0.yaml

```
Note: before you conduct a run with robust training, first pretrain the network with standard training to have a better initialization.
7. To assess the performance on the test set, run 
```
python test.py --run=./runs/voc_Nb=40_Ns=0_faster_rcnn_robust
```

## Citation
If you use this repository in your research, please cite
```
@article{bernhard2022robust_obj_det,
  title={Robust Object Detection in Remote Sensing Imagery with Noisy and Sparse Geo-Annotations (Full Version)},
  author={Bernhard, Maximilian and Schubert, Matthias},
  journal={arXiv preprint arXiv:2210.12989},
  year={2022}
}
```