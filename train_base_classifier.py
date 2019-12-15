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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ResNeXt, ResNet34
from utils import cal_parameters, get_dataset

name_dict = {'resnet': 'ResNet', 'resnext': 'ResNeXt'}


# test function (forward only)
def inference(net, data_loader, args):
    net.eval()
    correct = 0
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        # forward
        with torch.no_grad():
            output = net(x)

        # accuracy
        pred = output.max(1)[1]
        correct += float(pred.eq(y).sum())

    test_accuracy = correct / len(data_loader.dataset)
    return test_accuracy


# train function (forward, backward, update)
def train_epoch(net, data_loader, optimizer, args):
    net.train()
    loss_list = []
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        # forward
        output = net(x)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    return np.mean(loss_list)


def train(net, train_loader, test_loader, args):

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    best_train_loss = np.inf

    for epoch in range(args.epochs):
        if epoch in args.schedule:
            args.learning_rate *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate

        train_loss = train_epoch(net, train_loader, optimizer, args)
        print('Epoch: {}, training loss: {:.4f}.'.format(epoch + 1, train_loss))

        train_accuracy = inference(net, train_loader, args)
        print("Train accuracy: {:.4f}".format(train_accuracy))

        test_accuracy = inference(net, test_loader, args)
        print("Test accuracy: {:.4f}".format(test_accuracy))

        if train_loss < best_train_loss:
            best_train_loss = train_loss

            if args.model_name == 'resnext':
                save_name = 'ResNeXt{}_{}x{}d.pth'.format(args.depth, args.cardinality, args.base_width)
            elif args.model_name == 'resnet':
                save_name = 'ResNet34.pth'

            if use_cuda and args.n_gpu > 1:
                state = net.module.state_dict()
            else:
                state = net.state_dict()

            check_point = {'model_state': state, 'train_acc': train_accuracy, 'test_acc': test_accuracy}

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

    # Optimization options
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The Learning Rate.')
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

    parser.add_argument('--model_name', type=str, default='resnext', help='resnext or resnet')

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
    if args.model_name == 'resnext':
        net = ResNeXt(args.cardinality, args.depth, n_classes, args.base_width, args.widen_factor).to(args.device)
    elif args.model_name == 'resnet':
        net = ResNet34(n_classes=n_classes).to(args.device)
    else:
        print('Classifier {} not available.'.format(args.model_name))

    print('# Classifier parameters: ', cal_parameters(net))

    if use_cuda and args.n_gpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.n_gpu)))

    print('Dataset: {}'.format(args.dataset))
    train_data = get_dataset(data_name=args.dataset, data_dir=args.data_path, train=True, crop_flip=True)
    test_data = get_dataset(data_name=args.dataset, data_dir=args.data_path, train=False, crop_flip=False)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False)

    if args.inference:
        save_name = '{}{}_{}x{}d.pth'.format(name_dict[args.model_name], args.depth, args.cardinality, args.base_width)
        net.load_state_dict(torch.load(os.path.join(args.working_dir, save_name))['model_state'])
        acc = inference(net, test_loader, args)
        print('Test acc: {:.4f}'.format(acc))
    else:
        train(net, train_loader, test_loader, args)




