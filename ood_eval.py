from __future__ import print_function
import argparse
import os
import sys
import logging

import numpy as np
import hydra
from omegaconf import DictConfig


import torch
from torch.utils.data import DataLoader, Dataset


from models import resnet18, resnet34, resnet50
from sdim import SDIM
from utils import cal_parameters, get_dataset, AverageMeter


logger = logging.getLogger(__name__)


def get_model(name='resnet18', n_classes=10):
    """ get proper model from torchvision models. """
    model_list = ['resnet18', 'resnet34', 'resnet50']
    assert name in model_list, '{} not available, choose from {}'.format(name, model_list)

    classifier = eval(name)(n_classes=n_classes)
    return classifier


@hydra.main(config_path='configs/ood_config.yaml')
def run(args: DictConfig) -> None:
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = "cuda" if cuda_available and args.device == 'cuda' else "cpu"

    n_classes = args.get(args.dataset).n_classes
    rep_size = args.get(args.dataset).rep_size
    margin = args.get(args.dataset).margin

    classifier = get_model(name=args.classifier_name, n_classes=n_classes).to(args.device)

    sdim = SDIM(disc_classifier=classifier,
                n_classes=n_classes,
                rep_size=rep_size,
                mi_units=args.mi_units,
                margin=margin).to(args.device)

    base_dir = hydra.utils.to_absolute_path('logs/sdim/{}'.format(args.dataset))
    save_name = 'SDIM_{}.pth'.format(args.classifier_name)
    sdim.load_state_dict(torch.load(os.path.join(base_dir, save_name), map_location=lambda storage, loc: storage))

    ood_detection(sdim, args)


def ood_detection(sdim, args):
    """
    OOD samples detection, in_distribution: CIFAR10, out-distribution: SVHN.
    :param model: Pytorch model.
    :param hps: hyperparameters
    :return:
    """
    sdim.eval()

    # if args.problem == 'cifar10':
    #     out_problem = 'svhn'

    data_dir = hydra.utils.to_absolute_path(args.data_dir)

    threshold_list = []

    n_classes = args.get(args.dataset).n_classes
    for label_id in range(n_classes):
        # No data augmentation(crop_flip=False) when getting in-distribution thresholds
        dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, label_id=label_id, crop_flip=False)
        in_test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

        logger.info('Inference on {}, label_id {}'.format(args.dataset, label_id))
        in_ll_list = []
        for batch_id, (x, y) in enumerate(in_test_loader):
            x = x.to(args.device)
            y = y.to(args.device)
            ll = sdim(x)

            correct_idx = ll.argmax(dim=1) == y

            ll_, y_ = ll[correct_idx], y[correct_idx]  # choose samples are classified correctly
            in_ll_list += list(ll_[:, label_id].detach().cpu().numpy())

        thresh_idx = int(0.01 * len(in_ll_list))
        thresh = sorted(in_ll_list)[thresh_idx]
        print('threshold_idx/total_size: {}/{}, threshold: {:.3f}'.format(thresh_idx, len(in_ll_list), thresh))
        threshold_list.append(thresh)  # class mean as threshold

    print('Inference on {}'.format(args.ood_dataset))
    # eval on whole test set

    dataset = get_dataset(data_name=args.ood_dataset, data_dir=data_dir, train=False)
    out_test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

    reject_acc_dict = dict([(str(label_id), []) for label_id in range(n_classes)])

    for batch_id, (x, _) in enumerate(out_test_loader):
        x = x.to(args.device)
        ll = sdim(x)
        for label_id in range(n_classes):
            # samples whose ll lower than threshold will be successfully rejected.
            acc = (ll[:, label_id] < threshold_list[label_id]).float().mean().item()
            reject_acc_dict[str(label_id)].append(acc)

    logger.info('In-distribution dataset: {}, Out-distribution dataset: {}'.format(args.dataset, args.ood_dataset))
    rate_list = []
    for label_id in range(n_classes):
        acc = np.mean(reject_acc_dict[str(label_id)])
        rate_list.append(acc)
        logger.info('Label id: {}, reject success rate: {:.4f}'.format(label_id, acc))

    logger.info('Mean reject success rate: {:.4f}'.format(np.mean(rate_list)))


if __name__ == '__main__':
    run()
