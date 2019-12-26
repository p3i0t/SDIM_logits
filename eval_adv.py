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
from sdim_ce import SDIM
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

    name = 'SDIM_{}_{}.pth'.format(hps.classifier_name, hps.problem)
    checkpoint_path = os.path.join(hps.log_dir, name)
    sdim.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['model_state'])

    return sdim


def attack_run_rejection_policy(sdim, hps):
    """
    An attack run with rejection policy.
    :param model: Pytorch model.
    :param hps: hyperparameters
    :return:
    """
    sdim.eval()

    # Get thresholds
    print('Extracting thresholds.')
    threshold_list1 = []
    threshold_list2 = []
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

        thresh_idx = int(0.01 * len(in_ll_list))
        thresh1 = sorted(in_ll_list)[thresh_idx]
        thresh_idx = int(0.02 * len(in_ll_list))
        thresh2 = sorted(in_ll_list)[thresh_idx]
        threshold_list1.append(thresh1)  # class mean as threshold
        threshold_list2.append(thresh2)  # class mean as threshold
        print('1st & 2nd percentile thresholds: {:.3f}, {:.3f}'.format(thresh1, thresh2))

    attack_path = os.path.join(hps.attack_dir, hps.attack)
    if not os.path.exists(attack_path):
        os.mkdir(attack_path)

    thresholds1 = torch.tensor(threshold_list1).to(hps.device)
    thresholds2 = torch.tensor(threshold_list2).to(hps.device)

    if hps.attack == 'pgd':
        eps = 0.05
        hps.targeted = False
        adversary = LinfPGDAttack(
            sdim.disc_classifier, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=hps.targeted)
    elif hps.attack == 'cw':
        confidence = 0
        adversary = CarliniWagnerL2Attack(sdim.disc_classifier,
                                          num_classes=10,
                                          confidence=confidence,
                                          clip_min=0.,
                                          clip_max=1.,
                                          max_iterations=100
                                          )
    else:
        print('attack {} not available.'.format(hps.attack))

    # Evaluation
    n_correct_adv1 = 0   # total number of successfully classified adversarial examples
    n_correct_adv2 = 0   # total number of successfully classified adversarial examples
    n_wrong_adv1 = 0  # total number of incorrectly classified adversarial examples
    n_wrong_adv2 = 0  # total number of incorrectly classified adversarial examples
    n_rejected_adv1 = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv
    n_rejected_adv2 = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv
    n_eval = 0            # total number of correct classified samples by clean classifier

    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(hps.device), y.to(hps.device)

        # # Only evaluate on the correct classified samples by clean classifier.
        # with torch.no_grad():
        #     output = sdim.disc_classifier(x)
        # correct_idx = output.argmax(dim=1) == y
        # x, y = x[correct_idx], y[correct_idx]
        # n_eval += correct_idx.sum().item()
        n_eval += x.size()[0]

        # save_image(x, 'clean_x.png')
        # Generate adversarial examples
        adv_x = adversary.perturb(x, y)

        # save_image(adv_x, 'adv_x.png')
        # break
        # Prediction results with sdim-logit
        with torch.no_grad():
            output = sdim(adv_x)
        logits, preds = output.max(dim=1)

        # Eval on threshold 1st percentile
        rej_idx1 = logits < thresholds1[preds]
        n_rejected_adv1 += rej_idx1.sum().item()
        idx1 = rej_idx1.bitwise_not()

        n_correct_adv1 += (preds[idx1] == y[idx1]).sum().item()
        n_wrong_adv1 += (preds[idx1] != y[idx1]).sum().item()

        # Eval on threshold 2nd percentile
        rej_idx2 = logits < thresholds2[preds]
        n_rejected_adv2 += rej_idx2.sum().item()
        idx2 = rej_idx2.bitwise_not()

        n_correct_adv2 += (preds[idx2] == y[idx2]).sum().item()
        n_wrong_adv2 += (preds[idx2] != y[idx2]).sum().item()

    reject_rate1 = n_rejected_adv1 / n_eval
    reject_rate2 = n_rejected_adv2 / n_eval
    correct_rate1 = n_correct_adv1 / n_eval
    correct_rate2 = n_correct_adv2 / n_eval
    wrong_rate1 = n_wrong_adv1 / n_eval
    wrong_rate2 = n_wrong_adv2 / n_eval

    name = 'adv_results_{}_{}.pth'.format(hps.classifier_name, hps.attack)

    results_dict = {'reject_rate1': reject_rate1, 'reject_rate2': reject_rate2,
                    'correct_rate1': correct_rate1, 'correct_rate2': correct_rate2,
                    'wrong_rate1': wrong_rate1, 'wrong_rate2': wrong_rate2}
    torch.save(results_dict, os.path.join(hps.log_dir, name))

    print("================ Results =================")
    print('1st percentile, reject: {:.4f}, correct: {:.4f}, wrong: {:.4f}'.format(reject_rate1, correct_rate1, wrong_rate1))
    print('2nd percentile, reject: {:.4f}, correct: {:.4f}, wrong: {:.4f}'.format(reject_rate2, correct_rate2, wrong_rate2))


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
                        help="type of adversarial attack")
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
    sdim.disc_classifier.requires_grad = True

    attack_run_rejection_policy(sdim, hps)
