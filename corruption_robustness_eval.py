from __future__ import print_function
import os

import logging
import hydra
from omegaconf import DictConfig

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import resnet18, resnet34, resnet50
from sdim import SDIM
from utils import get_dataset


logger = logging.getLogger(__name__)

corruption_types = ['gaussian_noise', 'brightness', 'jpeg_compression', 'zoom_blur', 'gaussian_blur', 'defocus_blur',
                    'saturate', 'impulse_noise', 'snow', 'glass_blur', 'frost', 'fog', 'contrast', 'elastic_transform',
                    'pixelate', 'motion_blur', 'spatter', 'speckle_noise', 'shot_noise']


def get_model(name='resnet18', n_classes=10):
    """ get proper model from torchvision models. """
    model_list = ['resnet18', 'resnet34', 'resnet50']
    assert name in model_list,\
        '{} not available, choose from {}'.format(name, model_list)

    classifier = eval(name)(n_classes=n_classes)
    return classifier


def get_cifar_c_dataset(dir='CIFAR-10-C'):
    dir = hydra.utils.to_absolute_path(dir)  # change directory.
    from os import listdir
    files = [file for file in listdir(dir) if file != 'labels.npy' and file.endswith('.npy')]

    y = np.load(os.path.join(dir, 'labels.npy'))
    for file in files:
        file = os.path.join(dir, file)
        yield file.split('.')[0].split('/')[-1], np.load(file), y


class CorruptionDataset(Dataset):
    # for cifar10 and cifar100
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


def get_corruption_dataset(args, corruption_type, severity):
    assert severity in set(range(1, 5 + 1)), 'severity {} not available, choose from 1-5'.format(severity)
    corruption_data_dir = hydra.utils.to_absolute_path(args.get(args.dataset).corruption_data_dir)  # change directory.
    transform = transforms.ToTensor()
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        interval = 10000
        y = np.load(os.path.join(corruption_data_dir, 'labels.npy'))
        x = np.load(os.path.join(corruption_data_dir, '{}.npy'.format(corruption_type)))

        x_severity = x[(severity - 1) * interval: severity * interval]
        y_severity = y[(severity - 1) * interval: severity * interval]

        dataset = CorruptionDataset(x_severity, y_severity, transform=transform)
    elif args.dataset == 'tiny_imagenet':
        data_dir = os.path.join(corruption_data_dir, '{}/{}'.format(corruption_type, severity))
        dataset = datasets.ImageFolder(data_dir, transform=transform)

    return dataset


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

    results_dict0 = dict()
    results_dict1 = dict()
    results_dict2 = dict()
    samples_likelihood_dict = {}

    for corruption_type in corruption_types:
        for severity in range(1, 5 + 1):
            logger.info('==> Corruption type: {}, severity level: {}'.format(corruption_type, severity))
            dataset = get_corruption_dataset(args, corruption_type, severity)

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

                # if batch_id == 0:
                #     image_name = "sample_{}_severity{}.png".format(corruption_type, severity)
                #     idx = 0
                #     while idx < x.size(0):
                #         if target[idx] == pred[idx]:
                #             save_image(x[idx], image_name, normalize=True)
                #             samples_likelihood_dict[image_name] = values[idx]  # save sample's likelihood.
                #             break

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

            key = '{}_{}'.format(corruption_type, severity)
            acc_left0 = n_correct0 / (n_correct0 + n_false0)
            reject_rate0 = n_reject0 / n
            logger.info('no rejection, acc_left: {:.4f}, rejection_rate: {:.4f}'.format(acc_left0, reject_rate0))
            results_dict0[key] = {'acc_left': acc_left0, 'rejection_rate': reject_rate0}

            acc_left1 = n_correct1 / (n_correct1 + n_false1)
            reject_rate1 = n_reject1 / n
            logger.info('1st percentile, acc_left: {:.4f}, rejection_rate: {:.4f}'.format(acc_left1, reject_rate1))
            results_dict1[key] = {'acc_left': acc_left1, 'rejection_rate': reject_rate1}

            acc_left2 = n_correct2 / (n_correct2 + n_false2)
            reject_rate2 = n_reject2 / n
            logger.info('2nd percentile, acc_left: {:.4f}, rejection_rate: {:.4f}'.format(acc_left2, reject_rate2))
            results_dict2[key] = {'acc_left': acc_left2, 'rejection_rate': reject_rate2}

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

    thresholds1, thresholds2 = extract_thresholds(sdim, args)
    corruption_eval(sdim, args, thresholds1, thresholds2)


if __name__ == "__main__":
    run()
