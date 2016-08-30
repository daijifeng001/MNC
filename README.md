# Instance-aware Semantic Segmentation via Multi-task Network Cascades

By Jifeng Dai, Kaiming He, Jian Sun

This python version is re-implemented by [Haozhi Qi](https://github.com/Oh233) when he was an intern at Microsoft Research.

### Introduction

MNC is an instance-aware semantic segmentation system based on deep convolutional networks, which won the first place in COCO segmentation challenge 2015, and test at a fraction of a second per image. We decompose the task of instance-aware semantic segmentation into related sub-tasks, which are solved by multi-task network cascades (MNC) with shared features. The entire MNC network is trained end-to-end with error gradients across cascaded stages.


<img src='data/readme_img/example.png', width='800'>


MNC was initially described in a [CVPR 2016 oral paper](http://arxiv.org/abs/1512.04412).

This repository contains a python implementation of MNC, which is ~10% slower than the original matlab implementation.

This repository includes a bilinear RoI warping layer, which enables gradient back-propagation with respect to RoI coordinates.

### Misc.

This code has been tested on Linux (Ubuntu 14.04), using K40/Titan X GPUs.

The code is built based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn).

MNC is released under the MIT License (refer to the LICENSE file for details).


### Citing MNC

If you find MNC useful in your research, please consider citing:

    @inproceedings{dai2016instance,
        title={Instance-aware Semantic Segmentation via Multi-task Network Cascades},
        author={Dai, Jifeng and He, Kaiming and Sun, Jian},
        booktitle={CVPR},
        year={2016}
    }

### Main Results
                   | training data       | test data             | mAP^r@0.5   | mAP^r@0.7   | time (K40)    | time (Titian X)
-------------------|:-------------------:|:---------------------:|:-----------:|:-----------:|:-------------:|:-------------:|
MNC, VGG-16        | VOC 12 train        | VOC 12 val            | 65.0%       | 46.3%       | 0.42sec/img   | 0.33sec/img

### Installation guide

1. Clone the MNC repository:
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/daijifeng001/MNC.git
  ```
 
2. Install Python packages: `numpy`, `scipy`, `cython`, `python-opencv`, `easydict`, `yaml`.

3. Build the Cython modules and the gpu_nms, gpu_mask_voting modules by:
  ```Shell
  cd $MNC_ROOT/lib
  make
  ```

4. Install `Caffe` and `pycaffe` dependencies (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html) for official installation guide)

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # CUDNN is recommended in building to reduce memory footprint
  USE_CUDNN := 1
  ```

5. Build Caffe and pycaffe:
    ```Shell
    cd $MNC_ROOT/caffe-mnc
    # If you have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

### Demo

First, download the trained MNC model.
```Shell
./data/scripts/fetch_mnc_model.sh
``` 

Run the demo:
```Shell
cd $MNC_ROOT
./tools/demo.py
```
Result demo images will be stored to ```data/demo/```.

The demo performs instance-aware semantic segmentation with a trained MNC model (using VGG-16 net). The model is pre-trained on ImageNet, and finetuned on VOC 2012 train set with additional annotations from [SBD](https://9bc0b5eb4c18f1fc9a28517a91305702c68a10ae.googledrive.com/host/0ByUkob0WA1-NQi1sNlg4WkJQbTg/codes/SBD/download.html). The mAP^r of the model is 65.0% on VOC 2012 validation set. The test speed per image is ~0.33sec on Titian X and ~0.42sec on K40.

### Training

This repository contains code to **end-to-end** train MNC for instance-aware semantic segmentation, where gradients across cascaded stages are counted in training.

#### Preparation:

0. Run `./data/scripts/fetch_imagenet_models.sh` to download the ImageNet pre-trained VGG-16 net. 
0. Download the VOC 2007 dataset to ./data/VOCdevkit2007
0. Run `./data/scripts/fetch_sbd_data.sh` to download the VOC 2012 dataset together with the additional segmentation annotations in [SBD](https://9bc0b5eb4c18f1fc9a28517a91305702c68a10ae.googledrive.com/host/0ByUkob0WA1-NQi1sNlg4WkJQbTg/codes/SBD/download.html) to ./data/VOCdevkitSDS.

#### 1. End-to-end training of MNC for instance-aware semantic segmentation

To end-to-end train a 5-stage MNC model (on VOC 2012 train), use `experiments/scripts/mnc_5stage.sh`. Final mAP^r@0.5 should be ~65.0% (mAP^r@0.7 should be ~46.3%), on VOC 2012 validation.

```Shell
cd $MNC_ROOT
./experiments/scripts/mnc_5stage.sh [GPU_ID] VGG16 [--set ...]
# GPU_ID is the GPU you want to train on
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng 1701 RNG_SEED 1701
```

#### 2. Training of CFM for instance-aware semantic segmentation

The code also includes an entry to train a [convolutional feature masking](https://arxiv.org/abs/1412.1283) (CFM) model for instance aware semantic segmentation.

    @inproceedings{dai2015convolutional,
        title={Convolutional Feature Masking for Joint Object and Stuff Segmentation},
        author={Dai, Jifeng and He, Kaiming and Sun, Jian},
        booktitle={CVPR},
        year={2015}
    }

##### 2.1. Download pre-computed MCG proposals

Download and process the pre-computed MCG proposals.

```Shell
cd $MNC_ROOT
./data/scripts/fetch_mcg_data.sh
python ./tools/prepare_mcg_maskdb.py --para_job 24 --db train --output data/cache/voc_2012_train_mcg_maskdb/
python ./tools/prepare_mcg_maskdb.py --para_job 24 --db val --output data/cache/voc_2012_val_mcg_maskdb/
```
Resulting proposals would be at folder ```data/MCG/```.

##### 2.2. Train the model

Run `experiments/scripts/cfm.sh` to train on VOC 2012 train set. Final mAP^r@0.5 should be ~60.5% (mAP^r@0.7 should be ~42.6%), on VOC 2012 validation.

```Shell
cd $MNC_ROOT
./experiments/scripts/cfm.sh [GPU_ID] VGG16 [--set ...]
# GPU_ID is the GPU you want to train on
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng 1701 RNG_SEED 1701
```

#### 3. End-to-end training of Faster-RCNN for object detection

Faster-RCNN can be viewed as a 2-stage cascades composed of region proposal network (RPN) and object detection network. Run script `experiments/scripts/faster_rcnn_end2end.sh` to train a Faster-RCNN model on VOC 2007 trainval. Final mAP^b should be ~69.1% on VOC 2007 test.

```Shell
cd $MNC_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] VGG16 [--set ...]
# GPU_ID is the GPU you want to train on
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```
