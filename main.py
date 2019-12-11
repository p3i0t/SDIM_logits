from __future__ import print_function
import argparse
import os
import sys
import time

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import models
from sdim_ce import SDIM
from utils import cal_parameters, get_dataset, AverageMeter


def load_pretrained_model(hps):
    checkpoint_path = '{}_{}.pth'.format(hps.classifier_name, hps.problem)
    print('Load pre-trained checkpoint: {}'.format(checkpoint_path))
    pre_trained_dir = os.path.join(hps.log_dir, checkpoint_path)

    model = models.ResNet34(num_c=hps.n_classes)
    model.load_state_dict(torch.load(pre_trained_dir, map_location=lambda storage, loc: storage))
    return model


def train(sdim, optimizer, hps):
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    dataset = get_dataset(data_name=hps.problem, train=True)
    train_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_train, shuffle=True)

    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    results_dict = dict({'train_loss': [], 'train_MI': [], 'train_CE': [],
                         'test_loss': [], 'test_MI': [], 'test_CE': []})

    min_loss = 1e3
    for epoch in range(1, hps.epochs + 1):
        sdim.train()

        Timer = AverageMeter('timer')
        loss_meter = AverageMeter('loss')
        MI_meter = AverageMeter('MI')
        CE_meter = AverageMeter('CE')

        end = time.time()
        for batch_id, (x, y) in enumerate(train_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            optimizer.zero_grad()

            loss, mi_loss, ce_loss = sdim.eval_losses(x, y)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            MI_meter.update(mi_loss.item(), x.size(0))
            CE_meter.update(ce_loss.item(), x.size(0))

        Timer.update(time.time() - end)

        print('===> Epoch: {}'.format(epoch))
        print('loss: {:.4f}, MI: {:.4f}, CE: {:.4f}'.format(loss_meter.avg, MI_meter.avg, CE_meter.avg))

        results_dict['train_loss'].append(loss_meter)
        results_dict['train_MI'].append(MI_meter)
        results_dict['train_CE'].append(CE_meter)

        if loss_meter.avg < min_loss:
            min_loss = loss_meter.avg
            state = sdim.state_dict()

        # Evaluate accuracy on test set.
        sdim.eval()
        loss_meter = AverageMeter('loss')
        MI_meter = AverageMeter('MI')
        CE_meter = AverageMeter('CE')

        acc_meter = AverageMeter('Acc')
        for batch_id, (x, y) in enumerate(test_loader):
            x = x.to(hps.device)
            y = y.to(hps.device)

            preds = sdim(x).argmax(dim=1)
            acc = (preds == y).float().mean()
            acc_meter.update(acc.item(), x.size(0))

            loss, mi_loss, ce_loss = sdim.eval_losses(x, y)
            loss_meter.update(loss.item(), x.size(0))
            MI_meter.update(mi_loss.item(), x.size(0))
            CE_meter.update(ce_loss.item(), x.size(0))

        print('Test accuracy: {:.3f}'.format(acc_meter.avg))

        results_dict['test_loss'].append(loss_meter)
        results_dict['test_MI'].append(MI_meter)
        results_dict['test_CE'].append(CE_meter)

    name = 'SDIM_{}_{}.pth'.format(hps.classifier_name, hps.problem)
    checkpoint_path = os.path.join(hps.log_dir, name)
    checkpoint = {'results': results_dict, 'state': state}

    torch.save(checkpoint, checkpoint_path)
    print("Training time, total: {:.3f}s, epoch: {:.3f}s".format(Timer.sum, Timer.avg))


def inference(sdim, hps):
    sdim.eval()

    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    name = 'SDIM_{}_{}.pth'.format(hps.classifier_name, hps.problem)
    checkpoint_path = os.path.join(hps.log_dir, name)

    sdim.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['state'])

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
    print('Test accracy: {:.4f}'.format(np.mean(global_acc_list)))


def inference_rejection(sdim, hps):
    torch.manual_seed(hps.seed)
    np.random.seed(hps.seed)

    name = 'SDIM_{}_{}.pth'.format(hps.classifier_name, hps.problem)
    checkpoint_path = os.path.join(hps.log_dir, name)

    sdim.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['state'])
    sdim.eval()

    # Get thresholds
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

    # Evaluation
    dataset = get_dataset(data_name=hps.problem, train=False)
    test_loader = DataLoader(dataset=dataset, batch_size=hps.n_batch_test, shuffle=False)

    n_correct = 0
    n_false = 0
    n_reject = 0

    thresholds = torch.tensor(threshold_list).to(hps.device)
    result_str = ' & '.join('{:.1f}'.format(ll) for ll in threshold_list)
    print('thresholds: ', result_str)

    for batch_id, (x, target) in enumerate(test_loader):
        # Note that images are scaled to [-1.0, 1.0]
        x, target = x.to(hps.device), target.to(hps.device)

        with torch.no_grad():
            log_lik = sdim(x)

        values, pred = log_lik.max(dim=1)
        confidence_idx = values >= thresholds[pred]  # the predictions you have confidence in.
        reject_idx = values < thresholds[pred]       # the ones rejected.

        n_correct += pred[confidence_idx].eq(target[confidence_idx]).sum().item()
        n_false += (pred[confidence_idx] != target[confidence_idx]).sum().item()
        n_reject += reject_idx.float().sum().item()

    n = len(test_loader.dataset)
    acc = n_correct / n
    false_rate = n_false / n
    reject_rate = n_reject / n

    acc_remain = acc / (acc + false_rate)

    print('Test set:\nacc: {:.4f}, false rate: {:.4f}, reject rate: {:.4f}'.format(acc, false_rate, reject_rate))
    print('acc on remain set: {:.4f}'.format(acc_remain))
    return acc, reject_rate, acc_remain


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
                        default=32, help="output size of 1x1 conv network for mutual information estimation")
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

    classifier = load_pretrained_model(hps)
    sdim = SDIM(disc_classifier=classifier,
                n_classes=hps.n_classes,
                rep_size=hps.rep_size,
                mi_units=hps.mi_units,
                ).to(hps.device)
    optimizer = Adam(sdim.parameters(), lr=hps.lr)

    print('==>  # SDIM parameters: {}.'.format(cal_parameters(sdim)))

    if hps.inference:
        inference(sdim, hps)
    elif hps.rejection_inference:
        inference_rejection(sdim, hps)
    else:
        train(sdim, optimizer, hps)
