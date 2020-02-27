# SDIM-logits

This is the code repo for the paper *Reject Illegal Inputs with Generative Classifiers Derived from Any Discriminative Classifier*, submission for ECCV-2020. Part of the code is borrowed from open-sourced code [Deep Infomax (DIM)](https://github.com/rdevon/DIM).

## Repo Structure
```
sdim_logits
    |
    | -directories-
    | configs： configuration files storing the hyperparameters for training and evaluations.
    | data: directory for datasets.
    | logs: directory for checkpoints and evaluation results.
    | losses: various lower-bounds of MI as losses functions, borrowed from https://github.com/rdevon/DIM.
    | models: definitions of resnets.
    |
    | -files-
    | base_classifier_train.py: code for training base discriminative classifiers.
    | sdim_train.py: code for training sdim-logit models on base models.
    | sdim.py: definition of SDIM-logit framework.
    | mi_networks.py: definitions of Mutual Information evaluation networks, borrowed from https://github.com/rdevon/DIM
    | utils.py: some helper functions: get_dataset, AverageMeter, cal_parameters.
    | adv_robustness_eval.py: code for adversarial robustness evaluations.
    | corruption_robustness_eval.py: code for robustness evaluation on corrupted samples.
    | ood_eval.py: code for evaluation on out-of-distribution samples. 
    | cw_attack.py: implementation of CW attack with binary search.

```
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

If simply inference on the test set, add ``inference=True`` to the above commands.

See ``configs/base_config.yaml`` for the full training hyperparameters. 

### Train SDIM generative classifiers

Train SDIM-logit (ResNet18) on CIFAR10:

```python
python base_classifier_train.py dataset=cifar10 classifier_name=resnet18
```

Train SDIM-logit (ResNet18) on CIFAR100:

```python
python base_classifier_train.py dataset=cifar100 classifier_name=resnet18
```

Train SDIM-logit (ResNet18) on Tiny Imagenet:
```python
python base_classifier_train.py dataset=tiny_imagenet classifier_name=resnet18
```
Other available classifiers include ``resnet34, resnet50``. 

See ``configs/sdim_config.yaml`` for the full training hyperparameters. 

### Evaluation on Corrupted Samples
