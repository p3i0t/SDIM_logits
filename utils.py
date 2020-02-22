from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataset(data_name='cifar10', data_dir='data', train=True, label_id=None, crop_flip=True):
    """
    Get a dataset.
    :param data_name: str, name of dataset.
    :param data_dir: str, base directory of data.
    :param train: bool, return train set if True, or test set if False.
    :param label_id: None or int, return data with particular label_id.
    :param crop_flip: bool, whether use crop_flip as data augmentation.
    :return: pytorch dataset.
    """

    transform_3d_crop_flip = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((125.3/255, 123/255, 113.9/255), (63/255, 62.1/255, 66.7/255))
    ])

    transform_3d = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((125.3/255, 123/255, 113.9/255), (63/255, 62.1/255, 66.7/255))
    ])

    if train:
        # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
        transform = transform_3d if crop_flip is False else transform_3d_crop_flip
    else:
        transform = transform_3d

    if data_name == 'cifar10':
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    elif data_name == 'cifar100':
        dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
    elif data_name == 'svhn':
        split = 'train' if train else 'test'
        dataset = datasets.SVHN(data_dir, split=split, download=False, transform=transform)
    else:
        print('dataset {} is not available'.format(data_name))

    if label_id is not None:
        # select samples with particular label
        if data_name == 'cifar10' or data_name == 'cifar100':  #isinstance(dataset.targets, list):
            # for cifar10
            targets = np.array(dataset.targets)
            idx = targets == label_id
            dataset.targets = list(targets[idx])
            dataset.data = dataset.data[idx]
        elif data_name == 'svhn':
            idx = dataset.labels == label_id
            dataset.labels = dataset.labels[idx]
            dataset.data = dataset.data[idx]
    return dataset


def cal_parameters(model):
    """
    Calculate the number of parameters of a Pytorch model.
    :param model: torch.nn.Module
    :return: int, number of parameters.
    """
    return sum([para.numel() for para in model.parameters()])


def dataloader_test():
    for data_name in ['cifar10', 'cifar100', 'svhn']:
        print('Testing on {}'.format(data_name))
        for label_id in range(10):
            dataset = get_dataset(data_name=data_name, train=True, label_id=label_id, crop_flip=False)
            train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=False)

            for batch_id, (x, y) in enumerate(train_loader):
                assert (y == label_id).all(), 'label verification failed. dataset: {}, label_id: {}'.format(data_name, label_id)
                break
        print('Testing on {} passed !!!'.format(data_name))

    print('All dataloader testings passed !!!')


if __name__ == '__main__':
    dataloader_test()
