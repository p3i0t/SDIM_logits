from __future__ import print_function
import argparse
import os
import sys
import time

import logging
import hydra
from omegaconf import DictConfig


import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

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


def get_c_dataset(dir='data/CIFAR-10-C'):
    dir = hydra.utils.to_absolute_path(dir)  # change directory.
    from os import listdir
    files = [file for file in listdir(dir) if file != 'labels.npy']

    y = np.load(os.path.join(dir, 'labels.npy'))
    for file in files:
        file = os.path.join(dir, file)
        yield file.split('.')[0].split('/')[-1], np.load(file), y


class CorruptionDataset(Dataset):
    def __init__(self, x, y, transform=None):
        """

        :param x: numpy array
        :param y: numpy array
        """
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, item):
        sample = self.x[item]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.y[item]

    def __len__(self):
        return self.x.shape[0]



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


def corruption_eval(sdim, args, thresholds1, thresholds2):
    sdim.eval()
    thresholds0 = thresholds1 - 1e5   # set thresholds to be very low, so that no rejection happens.

    transform = transforms.ToTensor()
    interval = 10000

    results_dict0 = dict()
    results_dict1 = dict()
    results_dict2 = dict()
    samples_likelihood_dict = {}
    for corruption_id, (corruption_type, data, labels) in enumerate(get_c_dataset(args.data_dir)):
        logger.info('==> Corruption type: {}'.format(corruption_type))

        for severity in range(5):
            x_severity = data[severity * interval: (severity + 1) * interval]
            y_severity = labels[severity * interval: (severity + 1) * interval]

            dataset = CorruptionDataset(x_severity, y_severity, transform=transform)
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

                if batch_id == 0:
                    image_name = "sample_{}_severity{}.png".format(corruption_type, severity)
                    save_image(x[0], image_name, normalize=True)
                    assert target[0] == pred[0]
                    samples_likelihood_dict[image_name] = values[0]  # save sample's likelihood.

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
                n_correct0 += n_c 
                n_false0 += n_f 
                n_reject0 += n_r

                n_c, n_f, n_r = func(thresholds2)
                n_correct0 += n_c 
                n_false0 += n_f 
                n_reject0 += n_r

            n = len(test_loader.dataset)
            acc = n_correct / n
            false_rate = n_false / n
            reject_rate = n_reject / n

            acc_remain = acc / (acc + false_rate)

            key = '{}_{}'.format(corruption_type, severity + 1)
            results_dict0[key] = {'acc_left': n_correct0 / (n_correct0 + n_false0), 'rejection_rate': n_reject0 / n}
            results_dict1[key] = {'acc_left': n_correct1 / (n_correct1 + n_false1), 'rejection_rate': n_reject1 / n}
            results_dict2[key] = {'acc_left': n_correct2 / (n_correct2 + n_false2), 'rejection_rate': n_reject2 / n}

    torch.save(results_dict0, '{}_corruption_percentile0_results.pth'.format(args.classifier_name))
    torch.save(results_dict1, '{}_corruption_percentile1_results.pth'.format(args.classifier_name))
    torch.save(results_dict2, '{}_corruption_percentile2_results.pth'.format(args.classifier_name))


@hydra.main(config_path='configs/corruption_config.yaml')
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

    threshold1, threshold2 = extract_thresholds(sdim, args)
    corruption_eval(sdim, args, thresholds1, thresholds2)


if __name__ == "__main__":
    run()


