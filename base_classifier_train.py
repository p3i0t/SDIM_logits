# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ResNeXt, ResNet18
from utils import cal_parameters, get_dataset, AverageMeter

from advertorch.attacks import LinfPGDAttack

name_dict = {'resnet': 'ResNet', 'resnext': 'ResNeXt'}


def run_epoch(classifier, data_loader, args, optimizer=None, attack=None):
    """
    Run one epoch.
    :param classifier: torch.nn.Module representing the classifier.
    :param data_loader: dataloader
    :param args:
    :param optimizer: if None, then inference; if optimizer given, training and optimizing.
    :param attack: advertorch attack for adv examples generation. If None, do normal training or inference.
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

        if attack is not None:
            x_ = attack.perturb(x, y)
        else:
            x_ = x
        output = classifier(x_)
        loss = F.cross_entropy(output, y)

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        acc = (output.argmax(dim=1) == y).float().mean().item()
        acc_meter.update(acc, x.size(0))
    return loss_meter.avg, acc_meter.avg


def adv_train(classifier, train_loader, test_loader, args):
    eps = 0.02
    args.targeted = False
    adversary = LinfPGDAttack(
        classifier, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=50, eps_iter=0.01, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=args.targeted)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(10):
        # adversarial training
        adv_loss, adv_acc = run_epoch(classifier, train_loader, args, optimizer=optimizer, attack=adversary)
        print('Epoch: {}, Adv Train loss: {:.4f}, acc: {:.4f}'.format(epoch + 1, adv_loss, adv_acc))

        train_loss, train_acc = run_epoch(classifier, train_loader, args, optimizer=optimizer)
        print('Clean training loss: {:.4f}, acc: {:.4f}.'.format(train_loss, train_acc))

        # Eval on normal
        clean_loss, clean_acc = run_epoch(classifier, train_loader, args)
        print('Clean Test loss: {:.4f}, acc: {:.4f}'.format(clean_loss, clean_acc))

        # Eval on adv
        if epoch % 4 == 1:
            adv_loss, adv_acc = run_epoch(classifier, train_loader, args, attack=adversary)
            print('Adv Test loss: {:.4f}, acc: {:.4f}'.format(adv_loss, adv_acc))

        if args.classifier_name == 'resnext':
            save_name = 'AT_ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
        elif args.classifier_name == 'resnet':
            save_name = 'AT_ResNet18.pth'

        if use_cuda and args.n_gpu > 1:
            state = classifier.module.state_dict()
        else:
            state = classifier.state_dict()

        check_point = {'model_state': state, 'clean_acc': clean_acc, 'adv_acc': adv_acc}

        torch.save(check_point, os.path.join(args.working_dir, save_name))
        print("Saving new checkpoint ...")


def train(classifier, train_loader, test_loader, args):

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    best_train_loss = np.inf

    for epoch in range(args.epochs):
        if epoch in args.schedule:
            args.learning_rate *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate

        train_loss, train_acc = run_epoch(classifier, train_loader, args, optimizer=optimizer)
        print('Epoch: {}, training loss: {:.4f}, acc: {:.4f}.'.format(epoch + 1, train_loss, train_acc))

        test_acc = run_epoch(classifier, test_loader, args)
        print("Test acc: {:.4f}".format(test_acc))

        if train_loss < best_train_loss:
            best_train_loss = train_loss

            if args.classifier_name == 'resnext':
                save_name = 'ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
            elif args.classifier_name == 'resnet':
                save_name = 'ResNet18.pth'

            if use_cuda and args.n_gpu > 1:
                state = classifier.module.state_dict()
            else:
                state = classifier.state_dict()

            check_point = {'model_state': state, 'train_acc': train_acc, 'test_acc': test_acc}

            torch.save(check_point, os.path.join(args.working_dir, save_name))
            print("Saving new checkpoint ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNeXt or ResNet on CIFAR10',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--data_path', type=str, default='data', help='Root for the Cifar dataset.')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'svhn'], help='Choose between Cifar10/100.')

    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--adv_training", action="store_true",
                        help="Use adversarial training")

    # Optimization options
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.05, help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save',  type=str, default='./logs', help='Folder to save checkpoints.')
    parser.add_argument('--load',  type=str, default='./logs', help='Checkpoint path to resume / test.')

    # Architecture for resnext
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--n_gpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    parser.add_argument('--classifier_name', type=str, default='resnext', help='resnext or resnet')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()  # So error if typo

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    args.device = torch.device("cuda" if use_cuda else "cpu")
    print('device: ', args.device)

    n_classes = 10
    # Init checkpoints

    args.working_dir = os.path.join(args.save, args.dataset)

    if not os.path.isdir(args.working_dir):
        os.makedirs(args.working_dir)

    # Init model, criterion, and optimizer
    if args.classifier_name == 'resnext':
        classifier = ResNeXt(args.cardinality, args.depth, n_classes, args.base_width, args.widen_factor).to(args.device)
    elif args.classifier_name == 'resnet':
        classifier = ResNet18(n_classes=n_classes).to(args.device)
    else:
        print('Classifier {} not available.'.format(args.classifier_name))

    print('# Classifier parameters: ', cal_parameters(classifier))

    if use_cuda and args.n_gpu > 1:
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(args.n_gpu)))

    print('Dataset: {}'.format(args.dataset))
    train_data = get_dataset(data_name=args.dataset, data_dir=args.data_path, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=args.data_path, train=False, crop_flip=False)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False)

    if args.inference:
        if args.classifier_name == 'resnext':
            save_name = 'ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
        elif args.classifier_name == 'resnet':
            save_name = 'ResNet18.pth'
        classifier.load_state_dict(torch.load(os.path.join(args.working_dir, save_name))['model_state'])
        loss, acc = run_epoch(classifier, test_loader, args)
        print('Test loss: {:.4f}, acc: {:.4f}'.format(loss, acc))
    elif args.adv_training:
        # Perform adversarial training on pre-trained classifier.
        if args.classifier_name == 'resnext':
            save_name = 'ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
        elif args.classifier_name == 'resnet':
            save_name = 'ResNet18.pth'
        classifier.load_state_dict(torch.load(os.path.join(args.working_dir, save_name))['model_state'])
        adv_train(classifier, train_loader, test_loader, args)
    else:
        train(classifier, train_loader, test_loader, args)




