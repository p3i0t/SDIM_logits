# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""
import numpy as np

import hydra
from omegaconf import DictConfig
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
# from torchvision.models import resnet18, resnet34, resnet50
from models import resnet18, resnet34, resnet50
from utils import cal_parameters, get_dataset, AverageMeter


logger = logging.getLogger(__name__)


def get_model(name='resnet18', n_classes=10):
    """ get proper model from torchvision models. """
    model_list = ['resnet18', 'resnet34', 'resnet50']
    assert name in model_list, '{} not available, choose from {}'.format(name, model_list)

    classifier = eval(name)(n_classes=n_classes)
    return classifier


def run_epoch(classifier, data_loader, args, optimizer=None):
    """
    Run one epoch.
    :param classifier: torch.nn.Module representing the classifier.
    :param data_loader: dataloader
    :param args:
    :param optimizer: if None, then inference; if optimizer given, training and optimizing.
    :return: mean of loss, mean of accuracy of this epoch.
    """
    if optimizer:
        classifier.train()
    else:
        classifier.eval()

    loss_meter = AverageMeter('loss')
    acc_meter = AverageMeter('Acc')
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        output = classifier(x)
        loss = F.cross_entropy(output, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (output.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))
    return loss_meter.avg, acc_meter.avg


def train(classifier, train_loader, test_loader, args):
    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    best_train_loss = np.inf

    for epoch in range(1, args.epochs + 1):
        if epoch in args.schedule:
            args.learning_rate *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate

        train_loss, train_acc = run_epoch(classifier, train_loader, args, optimizer=optimizer)
        logger.info('Epoch: {}, training loss: {:.4f}, acc: {:.4f}.'.format(epoch, train_loss, train_acc))

        test_loss, test_acc = run_epoch(classifier, test_loader, args)
        logger.info("Test loss: {:.4f}, acc: {:.4f}".format(test_loss, test_acc))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            save_name = '{}.pth'.format(args.classifier_name)

            # # if use cuda and n_gpu > 1
            # if next(classifier.parameters()).is_cuda and args.n_gpu > 1:
            #     state = classifier.module.state_dict()
            # else:
            #     state = classifier.state_dict()
            state = classifier.state_dict()

            torch.save(state, save_name)
            logger.info("==> New optimal training loss & saving checkpoint ...")


@hydra.main(config_path='configs/base_config.yaml')
def run(args: DictConfig) -> None:
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = "cuda" if cuda_available and args.device == 'cuda' else "cpu"

    n_classes = args.get(args.dataset).n_classes
    if args.dataset == 'tiny_imagenet':
        args.epochs = 20
        args.learning_rate = 0.001
        classifier = eval('torchvision.models.' + args.classifier_name)(pretrained=True)
        classifier.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier.fc.out_features = 200
        classifier.to(device)
        
    else:
        classifier = get_model(name=args.classifier_name, n_classes=n_classes).to(device)

    # if device == 'cuda' and args.n_gpu > 1:
    #     classifier = torch.nn.DataParallel(classifier, device_ids=list(range(args.n_gpu)))

    logger.info('Base classifier name: {}, # parameters: {}'.format(args.classifier_name, cal_parameters(classifier)))

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    train_loader = DataLoader(dataset=train_data, batch_size=args.n_batch_train, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.n_batch_test, shuffle=False)

    if args.inference:
        save_name = '{}.pth'.format(args.classifier_name)
        classifier.load_state_dict(torch.load(save_name, map_location=lambda storage, loc: storage))
        loss, acc = run_epoch(classifier, test_loader, args)
        logger.info('Inference loss: {:.4f}, acc: {:.4f}'.format(loss, acc))
    else:
        train(classifier, train_loader, test_loader, args)


if __name__ == '__main__':
    run()

