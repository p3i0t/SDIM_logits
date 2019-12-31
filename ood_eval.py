from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack


from models import ResNeXt, ResNet18
from sdim import SDIM
from utils import cal_parameters, get_dataset, AverageMeter
from torchvision.utils import save_image


def load_pretrained_sdim(hps):
    # Init model, criterion, and optimizer
    if hps.classifier_name == 'resnext':
        classifier = ResNeXt(hps.cardinality, hps.depth, hps.n_classes, hps.base_width, hps.widen_factor).to(hps.device)
    elif hps.classifier_name == 'resnet':
        classifier = ResNet18(n_classes=hps.n_classes).to(hps.device)
    else:
        print('Classifier {} not available.'.format(hps.classifier_name))

    print('# Classifier parameters: ', cal_parameters(classifier))

    sdim = SDIM(disc_classifier=classifier,
                n_classes=hps.n_classes,
                rep_size=hps.rep_size,
                mi_units=hps.mi_units,
                ).to(hps.device)

    save_name = 'SDIM_{}_{}.pth'.format(hps.classifier_name, hps.problem)
    if hps.adv_training:
        save_name = 'AT_' + save_name

    checkpoint_path = os.path.join(hps.log_dir, save_name)
    sdim.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['model_state'])

    return sdim


def ood_detection(sdim, hps):
    """
    OOD samples detection, in_distribution: CIFAR10, out-distribution: SVHN.
    :param model: Pytorch model.
    :param hps: hyperparameters
    :return:
    """
    sdim.eval()

    if hps.problem == 'fashion':
        out_problem = 'mnist'

    elif hps.problem == 'cifar10':
        out_problem = 'svhn'

    threshold_list = []
    for label_id in range(hps.n_classes):
        # No data augmentation(crop_flip=False) when getting in-distribution thresholds
        dataset = get_dataset(data_name=hps.problem, train=True, label_id=label_id, crop_flip=False)
        in_test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

        print('Inference on {}, label_id {}'.format(hps.problem, label_id))
        in_ll_list = []
        for batch_id, (x, y) in enumerate(in_test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)
            ll = sdim(x)

            correct_idx = ll.argmax(dim=1) == y

            ll_, y_ = ll[correct_idx], y[correct_idx]  # choose samples are classified correctly
            in_ll_list += list(ll_[:, label_id].detach().cpu().numpy())

        thresh_idx = int(hps.percentile * len(in_ll_list))
        thresh = sorted(in_ll_list)[thresh_idx]
        print('threshold_idx/total_size: {}/{}, threshold: {:.3f}'.format(thresh_idx, len(in_ll_list), thresh))
        threshold_list.append(thresh)  # class mean as threshold

    print('Inference on {}'.format(out_problem))
    # eval on whole test set
    dataset = get_dataset(data_name=out_problem, train=False)
    out_test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    reject_acc_dict = dict([(str(label_id), []) for label_id in range(hps.n_classes)])

    for batch_id, (x, _) in enumerate(out_test_loader):
        x = x.to(hps.device)
        ll = sdim(x)
        for label_id in range(hps.n_classes):
            # samples whose ll lower than threshold will be successfully rejected.
            acc = (ll[:, label_id] < threshold_list[label_id]).float().mean().item()
            reject_acc_dict[str(label_id)].append(acc)

    print('==================== OOD Summary ====================')
    print('In-distribution dataset: {}, Out-distribution dataset: {}'.format(hps.problem, out_problem))
    rate_list = []
    for label_id in range(hps.n_classes):
        acc = np.mean(reject_acc_dict[str(label_id)])
        rate_list.append(acc)
        print('Label id: {}, reject success rate: {:.4f}'.format(label_id, acc))

    print('Mean reject success rate: {:.4f}'.format(np.mean(rate_list)))
    print('=====================================================')


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser(description='PyTorch Implementation of SDIM_logits.')
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--no_rejection", action="store_true",
                        help="Used in inference mode with rejection")
    parser.add_argument("--adv_training", action="store_true",
                        help="Use pre-trained classifier with adversarial training")
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
    parser.add_argument("--pixel_eps", type=int, default=2,
                        help="norm bound of pgd attack, i.e. number of pixels.")

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

    ood_detection(sdim, hps)
