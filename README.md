# SDIM-logits

This is the code repo for the paper *Reject Illegal Inputs with Generative Classifiers Derived from Any Discriminative Classifier*, submission for ECCV-2020. Part of the code is borrowed from open-sourced code [Deep Infomax (DIM)](https://github.com/rdevon/DIM).

## Usage

### Train base discriminative classifiers

Train ResNet18 on CIFAR10:

```python
python base_classifier_train.py dataset=cifar10 classifier_name=resnet18
```

Train ResNet18 on CIFAR100:

```python
python base_classifier_train.py dataset=cifar100 classifier_name=resnet18
```

Train ResNet18 on Tiny Imagenet (200 classes, 500 images of size 64x64 for each class; 50 images in val and test):
```python
python base_classifier_train.py dataset=tiny_imagenet classifier_name=resnet18
```
Other available classifiers include ``resnet34, resnet50``. 

See ``configs/base_config.yaml`` for the full training hyperparameters. 

### Train SDIM generative classifiers

See ``configs/sdim_config.yaml`` for the full training hyperparameters. 

### Evaluation on Corrupted Samples
