# Object-based-Illumination-Transfer

This is the official code of the paper Object-based Illumination Transferring and Rendering for Applications of Mixed Reality

# TODO
- [x] inference
- [x] train
- [x] test

# Getting Started

### Requirements

+ Ubuntu 16.04 LTS
+ CUDA 9.1
+ CUDNN 7.1.3
+ Python 3.7
+ Pytorch 1.1
+ other libs in `requirements.txt`, please run `pip install -r requirements.txt`

### Install

1. Create a conda virtual environment and activate it:
```
conda create -n illuminet python=3.7
source activate illuminet
```
2. install pytorch in [https://pytorch.org/](https://pytorch.org/). Select your preferences and run the recommended install command.
3. install others libs.
```
pip install -r requirements.txt
```
4. clone this repo:
```
git clone https://github.com/vr2021id2122/Object-based-Illumination-Transfer.git
```


### Trainning

#### Data Preparation
Put your mesh data into the `Data` folder. The `Data` folder Tree:
```
+-- light
|   +-- plane
|   |   +--train
|   |   +--val
|   |   +--test
|   +-- bunny
|   |   +--train
|   |   +--val
|   |   +--test
|   +-- ......
+-- color
|   +-- bunny
|   |   +--train
|   |   +--val
|   |   +--test
|   +-- ......
```

#### Training Steps
+ First, you need to train the GAE of two different domain objects separately by running the `trainGAE.py` file. Before that, you need to run `python -m visdom.server`.
+ Second, you need to train the `Light2Color` model of the target object by running the `trainPairGAE.py` file.
+ Finally, you train the GAN to finish the attribute transfer from object A to object B by running the `trainGAN.py` file.

Before training, you need to modify the `args` of `.py` file according to your own custom dataset name and other information.

### Testing
After modifying the path of the trained model and dataset in `args` in the `test.py` file, run it.

### Inference
After modifying the path of the trained model and your custom dataset in `args` in the `inference.py` file, run it.

### Tips
This framework is not only suitable for the transfer of lighting attributes from planar objects to target objects, but also can be extended to other objects, and can be used for the transfer of any three-dimensional object attributes (such as shape transfer, pose transfer etc.).
