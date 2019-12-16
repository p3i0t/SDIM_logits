import os
from os import listdir
import torch
import numpy as np


dir='data/CIFAR-10-C'

classifier_name = 'resnet'
log_dir = 'logs'
percentile = 0.01
save_path = os.path.join(log_dir, '{}_cifar10-c_per{}_results.pth'.format(classifier_name, int(100 * percentile)))
results_dict = torch.load(save_path)

corruption_types = [file.split('.')[0] for file in listdir(dir) if file != 'labels.npy']
print('classifier: {}, percentile: {}'.format(classifier_name, percentile))

acc_remain_list = []
reject_rate_list = []
for severity in range(1, 5 + 1):
    keys = ['{}_{}'.format(c_type, severity) for c_type in corruption_types]
    acc_remain = np.mean([results_dict[k]['acc_remain'] for k in keys])
    reject_rate = np.mean([results_dict[k]['rejection_rate'] for k in keys])

    acc_remain_list.append(acc_remain)
    reject_rate_list.append(reject_rate)

    print('severity {}, acc_remain: {:.4f}, rejection_rate: {:.4f}'.format(severity, acc_remain, reject_rate))

print('mean acc_ramain: {:.4f}, rejection_rate: {:.4f}'.format(np.mean(acc_remain_list), np.mean(reject_rate_list)))
