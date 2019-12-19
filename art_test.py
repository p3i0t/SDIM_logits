from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from advertorch.attacks import LinfPGDAttack


from models import ResNeXt, ResNet34
from sdim_ce import SDIM
from utils import cal_parameters, get_dataset, AverageMeter


def load_pretrained_sdim(hps):
    # Init model, criterion, and optimizer
    if hps.classifier_name == 'resnext':
        classifier = ResNeXt(hps.cardinality, hps.depth, hps.n_classes, hps.base_width, hps.widen_factor).to(hps.device)
    elif hps.classifier_name == 'resnet':
        classifier = ResNet34(n_classes=hps.n_classes).to(hps.device)
    else:
        print('Classifier {} not available.'.format(hps.classifier_name))

    print('# Classifier parameters: ', cal_parameters(classifier))

    sdim = SDIM(disc_classifier=classifier,
                n_classes=hps.n_classes,
                rep_size=hps.rep_size,
                mi_units=hps.mi_units,
                ).to(hps.device)

    name = 'SDIM_{}_{}.pth'.format(hps.classifier_name, hps.problem)
    checkpoint_path = os.path.join(hps.log_dir, name)
    sdim.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['model_state'])

    return sdim


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser(description='PyTorch Implementation of SDIM_logits.')
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--no_rejection", action="store_true",
                        help="Used in inference mode with rejection")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

    parser.add_argument("--attack_dir", type=str,
                        default='./attack_logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem cifar10 | svhn")
    parser.add_argument("--n_classes", type=int,
                        default=10, help="number of classes of dataset.")
    parser.add_argument("--data_dir", type=str, default='data',
                        help="Location of data")

    # Optimization hyperparams:
    parser.add_argument("--n_batch_train", type=int,
                        default=128, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=200, help="Minibatch size")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total number of training epochs")

    # Inference hyperparams:
    parser.add_argument("--percentile", type=float, default=0.01,
                        help="percentile value for inference with rejection.")

    # Architecture for resnext
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

    # sdim hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=32, help="Image size")
    parser.add_argument("--mi_units", type=int,
                        default=64, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=10, help="size of the global representation from encoder")
    parser.add_argument("--classifier_name", type=str, default='resnet',
                        help="classifier name: resnet|densenet")
    parser.add_argument("--attack", type=str, default='cw',
                        help="Location of data")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    # Create log dir
    if not os.path.exists(hps.log_dir):
        os.mkdir(hps.log_dir)

    if not os.path.exists(hps.attack_dir):
        os.mkdir(hps.attack_dir)

    sdim = load_pretrained_sdim(hps).to(hps.device)
    sdim.eval()

    dataset = get_dataset(data_name=hps.problem, train=False)
    # hps.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    attack = LinfPGDAttack(sdim)

    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        adv_x = attack.perturb(x)

        pred = sdim(adv_x).argmax(dim=1)

        acc = (pred == y).float().mean().item()
        print('acc: ', acc)
        break



