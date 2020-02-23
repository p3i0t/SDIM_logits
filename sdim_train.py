from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

#from models import ResNet18, ResNeXt
from models import resnet18, resnet34, resnet50
from sdim import SDIM
from utils import cal_parameters, get_dataset, AverageMeter


# name_dict = {'resnet': 'ResNet', 'resnext': 'ResNeXt'}


def get_model(name='resnet18', n_classes=10):
    """ get proper model from torchvision models. """
    model_list = ['resnet18', 'resnet34', 'resnet50']
    assert name in model_list, '{} not available, choose from {}'.format(name, model_list)

    classifier = eval(name)(n_classes=n_classes)
    return classifier


def load_pretrained_model(hps):
    classifier = get_model(name=hps.classifier_name, n_classes=hps.n_classes).to(hps.device)

    save_name = '{}.pth'.format(args.classifier_name)
    base_dir = 'logs/base/{}'.format(hps.problem)
    classifier.load_state_dict(torch.load(os.path.join(base_dir, save_name)))
    return classifier

    # # Init model, criterion, and optimizer
    # if hps.classifier_name == 'resnext':
    #     classifier = ResNeXt(hps.cardinality, hps.depth, hps.n_classes, hps.base_width, hps.widen_factor).to(hps.device)
    # elif hps.classifier_name == 'resnet':
    #     classifier = ResNet18(n_classes=hps.n_classes).to(hps.device)
    # else:
    #     print('Classifier {} not available.'.format(hps.classifier_name))

    # print('# Classifier parameters: ', cal_parameters(classifier))

    # if hps.classifier_name == 'resnext':
    #     save_name = 'ResNeXt{}_{}x{}d.pth'.format(hps.depth, hps.cardinality, hps.base_width)
    # elif hps.classifier_name == 'resnet':
    #     save_name = 'ResNet18.pth'


    # checkpoint = torch.load(os.path.join(hps.working_dir, save_name), map_location=lambda storage, loc: storage)
    # print('Load pre-trained checkpoint: {}'.format(save_name))

    # classifier.load_state_dict(checkpoint['model_state'])
    # return classifier


def train(sdim, optimizer, hps):
    sdim.disc_classifier.requires_grad = False
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    dataset = get_dataset(data_name=hps.problem, train=True, crop_flip=True)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)

    dataset = get_dataset(data_name=hps.problem, train=False, crop_flip=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    results_dict = dict({'train_loss': [], 'train_MI': [], 'train_nll': [], 'train_margin': [],
                         'test_loss': [], 'test_MI': [], 'test_nll': [], 'test_margin': []})

    min_loss = 1e3
    Timer = AverageMeter('timer')

    for epoch in range(1, hps.epochs + 1):
        sdim.train()

        loss_meter = AverageMeter('loss')
        MI_meter = AverageMeter('MI')
        nll_meter = AverageMeter('NLL')
        margin_meter = AverageMeter('margin')

        end = time.time()
        for batch_id, (x, y) in enumerate(train_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            optimizer.zero_grad()

            loss, mi_loss, nll_loss, ll_margin = sdim.eval_losses(x, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            MI_meter.update(mi_loss.item(), x.size(0))
            nll_meter.update(nll_loss.item(), x.size(0))
            margin_meter.update(ll_margin.item(), x.size(0))

        Timer.update(time.time() - end)

        print('===> Epoch: {}'.format(epoch))
        print('loss:{:.4f}, MI:{:.4f}, NLL:{:.4f}, margin:{:.4f}'.format(
            loss_meter.avg, MI_meter.avg, nll_meter.avg, margin_meter.avg))

        results_dict['train_loss'].append(loss_meter)
        results_dict['train_MI'].append(MI_meter)
        results_dict['train_nll'].append(nll_meter)
        results_dict['train_margin'].append(margin_meter)

        if loss_meter.avg < min_loss:
            min_loss = loss_meter.avg
            state = sdim.state_dict()

        # Evaluate accuracy on test set.
        sdim.eval()
        loss_meter = AverageMeter('loss')
        MI_meter = AverageMeter('MI')
        nll_meter = AverageMeter('NLL')
        margin_meter = AverageMeter('margin')

        acc_meter = AverageMeter('Acc')
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            with torch.no_grad():
                preds = sdim(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            acc_meter.update(acc.item(), x.size(0))

            with torch.no_grad():
                loss, mi_loss, nll_loss, ll_margin = sdim.eval_losses(x, y)

            loss_meter.update(loss.item(), x.size(0))
            MI_meter.update(mi_loss.item(), x.size(0))
            nll_meter.update(nll_loss.item(), x.size(0))
            margin_meter.update(ll_margin.item(), x.size(0))

        print('Test accuracy: {:.3f}'.format(acc_meter.avg))

        results_dict['test_loss'].append(loss_meter)
        results_dict['test_MI'].append(MI_meter)
        results_dict['test_nll'].append(nll_meter)
        results_dict['test_margin'].append(margin_meter)

    sdim_dir = 'logs/sdim/{}'.format(hps.problem)
    if not os.path.exists(hps.sdim_dir):
        os.mkdir(hps.sdim_dir)

    save_name = 'SDIM_{}.pth'.format(hps.classifier_name)

    checkpoint_path = os.path.join(hps.sdim_dir, save_name)
    checkpoint = {'results': results_dict, 'model_state': state}

    torch.save(checkpoint, checkpoint_path)
    print("Training time, total: {:.3f}s, epoch: {:.3f}s".format(Timer.sum, Timer.avg))


def inference(sdim, hps):
    sdim.eval()

    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    sdim_dir = 'logs/sdim/{}'.format(hps.problem)
    save_name = 'SDIM_{}.pth'.format(hps.classifier_name)

    checkpoint_path = os.path.join(hps.sdim_dir, save_name)

    sdim.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['model_state'])

    global_acc_list = []
    for label_id in range(hps.n_classes):
        dataset = get_dataset(data_name=hps.problem, train=False, label_id=label_id)
        test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

        acc_meter = AverageMeter('Acc')
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            preds = sdim(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            acc_meter.update(acc.item(), x.size(0))

        global_acc_list.append(acc_meter.avg)
        print('Class label {}, Test accuracy: {:.4f}'.format(label_id, acc_meter.avg))
    print('Test accuracy: {:.4f}'.format(np.mean(global_acc_list)))


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    import signal

    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser(description='PyTorch Implementation of SDIM_logits.')
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--inference", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--adv_training", action="store_true",
                        help="Use pre-trained classifier with adversarial training")
    parser.add_argument("--log_dir", type=str,
                        default='./logs', help="Location to save logs")

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

    # ResNeXt hyperparameters
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
                        help="classifier name: resnet | resnext")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    hps = parser.parse_args()  # So error if typo

    use_cuda = not hps.no_cuda and torch.cuda.is_available()

    torch.manual_seed(hps.seed)
    hps.device = torch.device("cuda" if use_cuda else "cpu")

    # # Create log dir
    # hps.working_dir = os.path.join(hps.log_dir, hps.problem)
    # if not os.path.exists(hps.working_dir):
    #     os.mkdir(hps.working_dir)

    classifier = load_pretrained_model(hps).to(hps.device)

    sdim = SDIM(disc_classifier=classifier,
                n_classes=hps.n_classes,
                rep_size=hps.rep_size,
                mi_units=hps.mi_units,
                ).to(hps.device)

    optimizer = Adam(sdim.parameters(), lr=hps.lr)

    print('==>  # SDIM parameters: {}.'.format(cal_parameters(sdim)))

    if hps.inference:
        inference(sdim, hps)
    else:
        train(sdim, optimizer, hps)
