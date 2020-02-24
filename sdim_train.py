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


def load_pretrained_model(args):
    """ load pretrained base discriminative classifier."""
    n_classes = args.get(args.dataset).n_classes
    classifier = get_model(name=args.classifier_name, n_classes=n_classes).to(args.device)
    save_name = '{}.pth'.format(args.classifier_name)
    base_dir = 'logs/base/{}'.format(args.dataset)
    path = hydra.utils.to_absolute_path(base_dir)
    classifier.load_state_dict(torch.load(os.path.join(path, save_name)))
    return classifier


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
    dataset = get_dataset(data_name=args.dataset, train=True, crop_flip=True)
    train_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_train, shuffle=True)

    dataset = get_dataset(data_name=args.dataset, train=False, crop_flip=False)
    test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

    results_dict = dict({'train_loss': [], 'train_MI': [], 'train_nll': [], 'train_margin': [], 'train_acc': [],
                         'test_loss': [], 'test_MI': [], 'test_nll': [], 'test_margin': [], 'test_acc': []})

    # specify log dir 
    writer = SummaryWriter('runs/sdim_train_{}_experiment'.format(args.dataset))

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
    torch.save(results, results_name)


@hydra.main(config_path='configs/sdim_config.yaml')
def run(args: DictConfig) -> None:
    cuda_available = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = "cuda" if cuda_available and args.device == 'cuda' else "cpu"

    n_classes = args.get(args.dataset).n_classes
    rep_size = args.get(args.dataset).rep_size
    margin = args.get(args.dataset).margin
    
    classifier = load_pretrained_model(args)

    sdim = SDIM(disc_classifier=classifier,
                n_classes=n_classes,
                rep_size=rep_size,
                mi_units=args.mi_units,
                margin=margin).to(args.device)

    optimizer = Adam(sdim.parameters(), lr=args.learning_rate)

    train(classifier, optimizer, args)

if __name__ == '__main__':
    run()