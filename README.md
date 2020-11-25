# Object-based-Illumination-Transfer

This is the official code of the paper with id 2122 submitted to IEEEVR. We are preparing a clean version of the complete code(including training code and inference code), which will be updated soon.

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

### install

1. Create a conda virtual environment and activate it:
```
conda create -n illuminet python=3.7
source activate illuminet
```
2. install pytorch in [https://pytorch.org/](https://pytorch.org/). Select your preferences and run the recommended install command.
3. install others libs
```
pip install -r requirements.txt
```
4. clone this repo:
```
git clone https://github.com/vr2021id2122/Object-based-Illumination-Transfer.git
```

### Data Preparation
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

### Trainning
+ First, you need to train the GAE of two different domain objects separately by running the `trainGAE.py` file. Before that, you need to run `python -m visdom.server`.
+ Second, you need to train the `Light2Color` model of the target object by running the `trainPairGAE.py` file.
+ Finally, you train the GAN to finish the attribute transfer from object A to object B by running the `trainGAN.py` file.

Before training, you need to modify the `args` of `.py` file according to your own custom dataset name and other information.

### Testing
After correcting the path of the trained model and dataset in `args` in the `test.py` file, run it.

### Inference
After correcting the path of the trained model and your custom dataset in `args` in the `inference.py` file, run it.
