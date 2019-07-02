# Pytorch Trainer For Kaggle

This repository comes from my codes for the "iMaterialist Challenge on Product Recognition" competition. Now it still has many shortcomings and is full of personal style.  I will continue to improve it, and hope you can find some useful information here.

## The Organization

```
Pytorch_Trainer/
|-- Data_Loader/
|   |--__init__.py
|   |-- ImageTransform.py
|   |-- ImageLoader.py
|   |-- ImageDataset.py
|
|--Ensemble/
|　　|--__init__.py
|　　|--ModelAVG.py
|
|-- LR_scheduler/
|   |-- CosineAnnealingWarmRestarts.py
|   |-- CyclicLR.py  
|   |-- Learing_Rate_Finder.py 
|   |-- __init__.py
|
|-- Pretrained_Models/
|   |-- TheModels
|   |-- __init__.py
|   |-- MyPretrainModels.py
|
|-- Trainer/
|   |-- __init__.py
|   |-- Trainer.py
|
|-- Utils/
|   |-- __init__.py
|   |-- filter_little_images.py
|   |-- compute_mean.py
|   |-- count_samplenumber_class.py
|
|-- main.py
|-- Data_config.txt
|-- README
```

### Data_Loader package:

Used to load train/val/test images. It consists of three part: ImageTransform, ImageLoader, ImageDataset. 

In ImageTransform, some image transforms are defined. They are used for data augement to avoid the overfeating.

 The ImageDataset is used to determine the image path of the training dataset and testing dataset.

Te ImageLoader is used to load the image in the path provided by dataset.

### Ensemble package:

ModelAVG: a simple implement of model averaging.

ToDo: ModelStacking

### LR_scheduler package:

LR_scheduler consists of 2 LR scheduler :

1. SGDR (CosineAnnealingWarmRestarts.py),
2. CyclicLR (CyclicLR.py).

also a lr estimator：

1. Learing_Rate_Finder

Acknowledgement:

The code of SGDR and CyclicLR are from PyTorch docs. And the learning_rate_finder is from FastAi blog.

### Pretrained_Models package:

Pretrained_Models consists many STO models, including ResNet, SENET, DenseNet and so on. And I also write a wrapper to load the pretrained model.

Acknowledgement:

All the prtrained models are from torchvision.models and 

[pretrained_model.pytorch]: https://github.com/Cadene/pretrained-models.pytorch

### Trainer package:

In this package, I create the Trainer class, which is used to train the model.

Now it does not have many features, just can train model ,validate model and save model.

Some other features like freeze_model, unfreeze_model will be included.

### Utils

Utils has some useful tools, like compute image means, std.

### Root package

main.py: the start of the project. 

Data_config.py: It is used to provide configs to the ImageDataset and ImageLoader.





