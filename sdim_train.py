from __future__ import print_function
import argparse
import os
import sys
import time
import hydra
from omegaconf import DictConfig
import logging

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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


def get_model_for_tiny_imagenet(name='resnet18', n_classes=200):
    classifier = eval('torchvision.models.' + name)(pretrained=True)
    classifier.avgpool = nn.AdaptiveAvgPool2d(1)
    classifier.fc = nn.Linear(classifier.fc.in_features, n_classes)
    return classifier


def load_pretrained_model(args):
    """ load pretrained base discriminative classifier."""
    n_classes = args.get(args.dataset).n_classes

    if args.dataset == 'tiny_imagenet':
        classifier = get_model_for_tiny_imagenet(name=args.classifier_name, n_classes=n_classes).to(args.device)
    else:
        classifier = get_model(name=args.classifier_name, n_classes=n_classes).to(args.device)
    if not args.infernce:
        save_name = '{}.pth'.format(args.classifier_name)
        base_dir = 'logs/base/{}'.format(args.dataset)
        path = hydra.utils.to_absolute_path(base_dir)
        classifier.load_state_dict(torch.load(os.path.join(path, save_name)))
    return classifier


def extract_thresholds(sdim, args):
    sdim.eval()
    # Get thresholds
    threshold_list1 = []
    threshold_list2 = []

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    for label_id in range(args.get(args.dataset).n_classes):
        # No data augmentation(crop_flip=False) when getting in-distribution thresholds
        dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, label_id=label_id, crop_flip=False)
        in_test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

        logger.info('Extracting thresholds on {}, label_id {}'.format(args.dataset, label_id))
        in_ll_list = []
        for batch_id, (x, y) in enumerate(in_test_loader):
            x = x.to(args.device)
            y = y.to(args.device)
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

    thresholds1 = torch.tensor(threshold_list1).to(args.device)
    thresholds2 = torch.tensor(threshold_list2).to(args.device)
    return thresholds1, thresholds2


def clean_eval(sdim, args, thresholds1, thresholds2):
    sdim.eval()
    thresholds0 = thresholds1 - 1e5   # set thresholds to be very low, so that no rejection happens.

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)
    test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False, num_workers=4)

    n_correct0, n_false0, n_reject0 = 0, 0, 0
    n_correct1, n_false1, n_reject1 = 0, 0, 0
    n_correct2, n_false2, n_reject2 = 0, 0, 0

    for batch_id, (x, target) in enumerate(test_loader):
        # Note that images are scaled to [-1.0, 1.0]
        x, target = x.to(args.device), target.long().to(args.device)

        with torch.no_grad():
            log_lik = sdim(x)

        values, pred = log_lik.max(dim=1)

        def func(thresholds):
            confidence_idx = values >= thresholds[pred]  # the predictions you have confidence in.
            reject_idx = values < thresholds[pred]       # the ones rejected.

            n_correct = pred[confidence_idx].eq(target[confidence_idx]).sum().item()
            n_false = (pred[confidence_idx] != target[confidence_idx]).sum().item()
            n_reject = reject_idx.float().sum().item()
            return n_correct, n_false, n_reject

        # Calculate
        n_c, n_f, n_r = func(thresholds0)
        n_correct0 += n_c
        n_false0 += n_f
        n_reject0 += n_r

        n_c, n_f, n_r = func(thresholds1)
        n_correct1 += n_c
        n_false1 += n_f
        n_reject1 += n_r

        n_c, n_f, n_r = func(thresholds2)
        n_correct2 += n_c
        n_false2 += n_f
        n_reject2 += n_r

    n = len(test_loader.dataset)

    acc_left0 = n_correct0 / (n_correct0 + n_false0)
    reject_rate0 = n_reject0 / n
    logger.info('no rejection, acc_left: {:.4f}, rejection_rate: {:.4f}'.format(acc_left0, reject_rate0))
    results_dict0 = {'acc_left': acc_left0, 'rejection_rate': reject_rate0}

    acc_left1 = n_correct1 / (n_correct1 + n_false1)
    reject_rate1 = n_reject1 / n
    logger.info('1st percentile, acc_left: {:.4f}, rejection_rate: {:.4f}'.format(acc_left1, reject_rate1))
    results_dict1 = {'acc_left': acc_left1, 'rejection_rate': reject_rate1}

    acc_left2 = n_correct2 / (n_correct2 + n_false2)
    reject_rate2 = n_reject2 / n
    logger.info('2nd percentile, acc_left: {:.4f}, rejection_rate: {:.4f}'.format(acc_left2, reject_rate2))
    results_dict2 = {'acc_left': acc_left2, 'rejection_rate': reject_rate2}

    torch.save(results_dict0, '{}_clean_percentile0_results.pth'.format(args.classifier_name))
    torch.save(results_dict1, '{}_clean_percentile1_results.pth'.format(args.classifier_name))
    torch.save(results_dict2, '{}_clean_percentile2_results.pth'.format(args.classifier_name))


def run_epoch(sdim, data_loader, args, optimizer=None):
    """
    Run one epoch.
    :param sdim: torch.nn.Module representing the sdim.
    :param data_loader: dataloader
    :param args:
    :param optimizer: if None, then inference; if optimizer given, training and optimizing.
    :return: mean of loss, mean of accuracy of this epoch.
    """
    if optimizer:
        sdim.train()
    else:
        sdim.eval()

    loss_meter = AverageMeter('Loss')
    MI_meter = AverageMeter('MI')
    nll_meter = AverageMeter('NLL')
    margin_meter = AverageMeter('Margin')
    acc_meter = AverageMeter('Acc')

    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)
        loss, mi_loss, nll_loss, ll_margin = sdim.eval_losses(x, y)
        loss_meter.update(loss.item(), x.size(0))

        MI_meter.update(mi_loss.item(), x.size(0))
        nll_meter.update(nll_loss.item(), x.size(0))
        margin_meter.update(ll_margin.item(), x.size(0))
            
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            preds = sdim(x).argmax(dim=1)
        acc = (preds == y).float().mean()
        acc_meter.update(acc.item(), x.size(0))

    return loss_meter.avg, MI_meter.avg, nll_meter.avg, margin_meter.avg, acc_meter.avg


def train(sdim, optimizer, args):
    sdim.disc_classifier.requires_grad = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=True, crop_flip=True)
    train_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_train, shuffle=True)

    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)
    test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

    results_dict = dict({'train_loss': [], 'train_MI': [], 'train_nll': [], 'train_margin': [], 'train_acc': [],
                         'test_loss': [], 'test_MI': [], 'test_nll': [], 'test_margin': [], 'test_acc': []})

    # specify log dir 
    writer_path = hydra.utils.to_absolute_path('runs/sdim_train_{}_experiment'.format(args.dataset))
    writer = SummaryWriter(writer_path)

    min_loss = 1e3
    for epoch in range(1, args.epochs + 1):
        # train epoch
        train_loss, train_mi, train_nll, train_margin, train_acc = run_epoch(sdim, train_loader, args, optimizer=optimizer)
        logger.info('Epoch: {}'.format(epoch))
        logger.info('training loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, margin: {:.4f}, acc: {:.4f}.'.format(train_loss, train_mi, train_nll, train_margin, train_acc))

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_mi', train_mi, epoch)
        writer.add_scalar('train_nll', train_nll, epoch)
        writer.add_scalar('train_margin', train_margin, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

        # save results
        results_dict['train_loss'].append(train_loss)
        results_dict['train_MI'].append(train_mi)
        results_dict['train_nll'].append(train_nll)
        results_dict['train_margin'].append(train_margin)
        results_dict['train_acc'].append(train_acc)

        # test epoch
        test_loss, test_mi, test_nll, test_margin, test_acc = run_epoch(sdim, test_loader, args)
        logger.info('testing loss: {:.4f}, mi: {:.4f}, nll: {:.4f}, margin: {:.4f}, acc: {:.4f}.'.format(test_loss, test_mi, test_nll, test_margin, test_acc))

        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_mi', test_mi, epoch)
        writer.add_scalar('test_nll', test_nll, epoch)
        writer.add_scalar('test_margin', test_margin, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        # save results
        results_dict['test_loss'].append(test_loss)
        results_dict['test_MI'].append(test_mi)
        results_dict['test_nll'].append(test_nll)
        results_dict['test_margin'].append(test_margin)
        results_dict['test_acc'].append(test_acc)

        # checkpoint
        if train_loss < min_loss:
            min_loss = train_loss
            state = sdim.state_dict()

            state_name = 'SDIM_{}.pth'.format(args.classifier_name)
            torch.save(state, state_name)

    results_name = 'SDIM_{}_results.pth'.format(args.classifier_name)
    torch.save(results_dict, results_name)


@hydra.main(config_path='configs/sdim_config.yaml')
def run(args: DictConfig) -> None:
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = "cuda" if cuda_available and args.device == 'cuda' else "cpu"

    n_classes = args.get(args.dataset).n_classes
    rep_size = args.get(args.dataset).rep_size
    margin = args.get(args.dataset).margin

    classifier = load_pretrained_model(args)
    if args.dataset == 'tiny_imagenet':
        args.data_dir = 'tiny_imagenet'

    sdim = SDIM(disc_classifier=classifier,
                n_classes=n_classes,
                rep_size=rep_size,
                mi_units=args.mi_units,
                margin=margin).to(args.device)

    optimizer = Adam(sdim.parameters(), lr=args.learning_rate)

    if args.inference:
        save_name = 'SDIM_{}.pth'.format(args.classifier_name)
        sdim.load_state_dict(torch.load(save_name, map_location=lambda storage, loc: storage))

        thresholds1, thresholds2 = extract_thresholds(sdim, args)
        clean_eval(sdim, args, thresholds1, thresholds2)
    else:
        train(sdim, optimizer, args)


if __name__ == '__main__':
    run()
