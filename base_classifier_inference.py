from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np

import torch
from torch.utils.data import DataLoader
import models
from utils import get_dataset, AverageMeter


def load_pretrained_model(hps):
    checkpoint_path = '{}_{}.pth'.format(hps.classifier_name, hps.problem)
    print('Load pre-trained checkpoint: {}'.format(checkpoint_path))
    pre_trained_dir = os.path.join(hps.log_dir, checkpoint_path)

    model = models.ResNet34(num_c=hps.n_classes)
    model.load_state_dict(torch.load(pre_trained_dir, map_location=lambda storage, loc: storage))
    return model


def inference(model, hps):
    model.eval()

    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    acc_meter = AverageMeter('Acc')
    for batch_id, (x, y) in enumerate(test_loader):
        x = x.to(hps.device)
        y = y.to(hps.device)

        preds = model(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_meter.update(acc.item(), x.size(0))

    print('Test accracy: {:.4f}'.format(np.mean(acc_meter.avg)))


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser(description='PyTorch Implementation of SDIM_logits.')
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--rejection_inference", action="store_true",
                        help="Used in inference mode with rejection")
    parser.add_argument("--ood_inference", action="store_true",
                        help="Used in ood inference mode")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem cifar10|svhn")
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

    # sdim hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=32, help="Image size")
    parser.add_argument("--mi_units", type=int,
                        default=64, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=10, help="size of the global representation from encoder")
    parser.add_argument("--classifier_name", type=str, default='resnet',
                        help="classifier name: resnet|densenet")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)

    hps.device = torch.device("cuda" if use_cuda else "cpu")

    # Create log dir
    logdir = os.path.abspath(hps.log_dir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    classifier = load_pretrained_model(hps).to(hps.device)