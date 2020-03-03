import argparse
import sys
import os
import logging
import hydra
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import resnet18, resnet34, resnet50
from sdim import SDIM

from advertorch.attacks import LinfPGDAttack, GradientSignAttack
from cw_attack import CW
from utils import cal_parameters, get_dataset, AverageMeter


logger = logging.getLogger(__name__)


def get_model(name='resnet18', n_classes=10):
    """ get proper model from torchvision models. """
    model_list = ['resnet18', 'resnet34', 'resnet50']
    assert name in model_list, '{} not available, choose from {}'.format(name, model_list)

    classifier = eval(name)(n_classes=n_classes)
    return classifier


@hydra.main(config_path='configs/adv_config.yaml')
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

    if args.sample_likelihood:
        sample_cases(sdim, args)
    else:
        if args.attack == 'pgd':
            pgd_attack(sdim, args)
        elif args.attack == 'fgsm':
            fgsm_attack(sdim, args)
        elif args.attack == 'cw':
            cw_attack(sdim, args)


# def attack_run(model, adversary, args):
#     model.eval()
#     dataset = get_dataset(data_name=args.dataset, train=False)
#     # args.n_batch_test = 1
#     test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

#     test_clnloss = 0
#     clncorrect = 0
#     test_advloss = 0
#     advcorrect = 0

#     attack_path = os.path.join(args.attack_dir, args.attack)
#     if not os.path.exists(attack_path):
#         os.mkdir(attack_path)

#     for batch_id, (clndata, target) in enumerate(test_loader):
#         # Note that images are scaled to [-1.0, 1.0]
#         clndata, target = clndata.to(args.device), target.to(args.device)
#         path = os.path.join(attack_path, 'original_{}.png'.format(batch_id))
#         save_image(clndata, path, normalize=True)

#         with torch.no_grad():
#             output = model(clndata)

#         print('original logits ', output.detach().cpu().numpy())
#         test_clnloss += F.cross_entropy(
#             output, target, reduction='sum').item()
#         pred = output.max(1, keepdim=True)[1]
#         #print('pred: ', pred)
#         clncorrect += pred.eq(target.view_as(pred)).sum().item()

#         advdata = adversary.perturb(clndata, target)
#         path = os.path.join(attack_path, '{}perturbed_{}.png'.format(prefix, batch_id))
#         save_image(advdata, path, normalize=True)

#         with torch.no_grad():
#             output = model(advdata)
#         print('adv logits ', output.detach().cpu().numpy())

#         test_advloss += F.cross_entropy(
#             output, target, reduction='sum').item()
#         pred = output.max(1, keepdim=True)[1]
#         #print('pred: ', pred)
#         advcorrect += pred.eq(target.view_as(pred)).sum().item()

#         #if batch_id == 2:
#         #    exit(0)
#         break

#     test_clnloss /= len(test_loader.dataset)
#     print('Test set: avg cln loss: {:.4f},'
#           ' cln acc: {}/{}'.format(
#         test_clnloss, clncorrect, len(test_loader.dataset)))

#     test_advloss /= len(test_loader.dataset)
#     print('Test set: avg adv loss: {:.4f},'
#           ' adv acc: {}/{}'.format(
#         test_advloss, advcorrect, len(test_loader.dataset)))

#     cln_acc = clncorrect / len(test_loader.dataset)
#     adv_acc = advcorrect / len(test_loader.dataset)
#     return cln_acc, adv_acc


def sample_cases(sdim, args):
    sdim.eval()
    n_classes = args.get(args.dataset).n_classes

    sample_likelihood_dict = {}
    # logger.info('==> Corruption type: {}, severity level {}'.format(corruption_type, level))
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False, crop_flip=False)

    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    x, y = next(iter(test_loader))
    x, y = x.to(args.device), y.long().to(args.device)

    def f_forward(x_, y_, image_name):
        with torch.no_grad():
            log_lik = sdim(x_)
        save_name = '{}.png'.format(image_name)
        save_image(x_, save_name, normalize=True)
        return log_lik[:, y_].item()

    sample_likelihood_dict['original'] = f_forward(x, y, 'original')

    eps_2 = 2 / 255
    eps_4 = 4 / 255
    eps_8 = 8 / 255

    x_u_4 = (x + torch.FloatTensor(x.size()).uniform_(-eps_4, eps_4).to(args.device)).clamp_(0., 1.)
    x_g_4 = (x + torch.randn(x.size()).clamp_(-eps_4, eps_4).to(args.device)).clamp_(0., 1.)
    x_u_8 = (x + torch.FloatTensor(x.size()).uniform_(-eps_8, eps_8).to(args.device)).clamp_(0., 1.)
    x_g_8 = (x + torch.randn(x.size()).clamp_(-eps_8, eps_8).to(args.device)).clamp_(0., 1.)

    sample_likelihood_dict['uniform_4'] = f_forward(x_u_4, y, 'uniform_4')
    sample_likelihood_dict['uniform_8'] = f_forward(x_u_8, y, 'uniform_8')
    sample_likelihood_dict['gaussian_4'] = f_forward(x_g_4, y, 'gaussian_4')
    sample_likelihood_dict['gaussian_8'] = f_forward(x_g_8, y, 'gaussian_8')

    adversary = LinfPGDAttack(
        sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_2,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    adv_pgd_2 = adversary.perturb(x, y)

    adversary = LinfPGDAttack(
        sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_4,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    adv_pgd_4 = adversary.perturb(x, y)

    adversary = LinfPGDAttack(
        sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps_8,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
        clip_max=1.0, targeted=False)

    adv_pgd_8 = adversary.perturb(x, y)

    adversary = CW(sdim, n_classes, max_iterations=1000, c=1, clip_min=0., clip_max=1., learning_rate=0.01,
                   targeted=False)

    adv_cw_1, _, _, _ = adversary.perturb(x, y)

    adversary = CW(sdim, n_classes, max_iterations=1000, c=10, clip_min=0., clip_max=1., learning_rate=0.01,
                   targeted=False)

    adv_cw_10, _, _, _ = adversary.perturb(x, y)

    sample_likelihood_dict['pgd_2'] = f_forward(adv_pgd_2, y, 'pgd_2')
    sample_likelihood_dict['pgd_4'] = f_forward(adv_pgd_4, y, 'pgd_4')
    sample_likelihood_dict['pgd_8'] = f_forward(adv_pgd_8, y, 'pgd_8')
    sample_likelihood_dict['cw_1'] = f_forward(adv_cw_1, y, 'cw_1')
    sample_likelihood_dict['cw_10'] = f_forward(adv_cw_10, y, 'cw_10')

    print(sample_likelihood_dict)
    save_dir = hydra.utils.to_absolute_path('attack_logs/case_study')
    torch.save(sample_likelihood_dict, os.path.join(save_dir, 'sample_likelihood_dict.pt'))


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


def adv_eval_with_rejection(sdim, adversary, args, thresholds1, thresholds2):
    """
    An attack run with rejection policy.
    :param sdim: Pytorch model.
    :param adversary: Advertorch adversary.
    :param args: hyperparameters
    :return:
    """
    sdim.eval()

    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    # Evaluation
    dataset = get_dataset(data_name=args.dataset, data_dir=data_dir, train=False)
    # args.n_batch_test = 1
    test_loader = DataLoader(dataset=dataset, batch_size=args.n_batch_test, shuffle=False)

    n_correct = 0   # total number of correct classified samples by clean classifier
    n_successful_adv = 0  # total number of successful adversarial examples generated
    n_rejected_adv1 = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv
    n_rejected_adv2 = 0   # total number of successfully rejected (successful) adversarial examples, <= n_successful_adv

    n_classes = args.get(args.dataset).n_classes
    l2_distortion_list = []
    for batch_id, (x, y) in enumerate(test_loader):
        # Note that images are scaled to [0., 1.0]
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            output = sdim(x)

        pred = output.argmax(dim=1)
        correct_idx = pred == y
        x, y = x[correct_idx], y[correct_idx]  # Only evaluate on the correct classified samples by clean classifier.
        n_correct += correct_idx.sum().item()

        target = ((y + np.random.randint(n_classes)) % n_classes).long()

        if batch_id == 0:
            logger.info('correct labels {}'.format(y[:8]))
            logger.info('attacked labels {}'.format(target[:8]))

        if args.attack == 'cw':
            adv_x, l2_dist, adv_loss, loss = adversary.perturb(x, target)
        else:
            adv_x = adversary.perturb(x, target)

        with torch.no_grad():
            output = sdim(adv_x)

        pred = output.argmax(dim=1)
        successful_idx = pred == target   # idx of successful adversarial examples.

        adv_x = adv_x[successful_idx]
        x = x[successful_idx]
        y = y[successful_idx]
        target = target[successful_idx]

        values, pred = output[successful_idx].max(dim=1)

        # cal for successful ones.
        if args.attack == 'cw':
            l2_distortion = l2_dist.mean().item()
        else:
            diff = adv_x - x
            l2_distortion = diff.norm(p=2, dim=-1).mean().item()  # mean l2 distortion

        if batch_id == 0:
            base_dir = hydra.utils.to_absolute_path('imgs')
            if args.attack != 'cw':
                save_image(x[:8],os.path.join(base_dir, "normal_{}_eps{}.png".format(args.attack, adversary.eps)), normalize=True)
                save_image(adv_x[:8], os.path.join(base_dir, "adv_{}_eps{}.png".format(args.attack, adversary.eps)), normalize=True)
            logger.info('correct labels {}'.format(y[:8]))
            logger.info('attacked labels {}'.format(pred[:8]))

        # confidence_idx = values >= thresholds[pred]
        reject_idx1 = values < thresholds1[pred]  # idx of successfully rejected samples.
        reject_idx2 = values < thresholds2[pred]  # idx of successfully rejected samples.

        # adv_correct += pred[confidence_idx].eq(y[confidence_idx]).sum().item()
        n_successful_adv += successful_idx.float().sum().item()
        n_rejected_adv1 += reject_idx1.float().sum().item()
        n_rejected_adv2 += reject_idx2.float().sum().item()

        l2_distortion_list.append(l2_distortion)
        if batch_id % 10 == 0:
            logger.info('Evaluating on {}-th batch ...'.format(batch_id + 1))

    n = len(test_loader.dataset)
    reject_rate1 = n_rejected_adv1 / n_successful_adv
    reject_rate2 = n_rejected_adv2 / n_successful_adv
    success_adv_rate = n_successful_adv / n_correct

    l2_distortion = np.mean(l2_distortion_list)
    logger.info('Test set, clean classification accuracy: {}/{}={:.4f}'.format(n_correct, n, n_correct / n))
    logger.info('success rate of adv examples generation: {}/{}={:.4f}'.format(n_successful_adv, n_correct, success_adv_rate))
    logger.info('Mean L2 distortion of Adv Examples: {:.4f}'.format(l2_distortion))
    logger.info('1st percentile, reject success rate: {}/{}={:.4f}'.format(n_rejected_adv1, n_successful_adv, reject_rate1))
    logger.info('2nd percentile, reject success rate: {}/{}={:.4f}'.format(n_rejected_adv2, n_successful_adv, reject_rate2))

    return l2_distortion, reject_rate1, reject_rate2


def fgsm_attack(sdim, args):
    thresholds1, thresholds2 = extract_thresholds(sdim, args)
    eps_list = [0.01, 0.02, 0.05, 0.1]  # same as baseline DeepBayes

    results_dict = {'reject_rate1': [], 'reject_rate2': [], 'l2_distortion': []}
    for eps in eps_list:
        adversary = GradientSignAttack(
            sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            clip_min=0.,
            clip_max=1.,
            targeted=args.targeted
        )
        logger.info('epsilon = {:.4f}'.format(adversary.eps))
        l2_dist, rj_rate1, rj_rate2 = adv_eval_with_rejection(sdim, adversary, args, thresholds1, thresholds2)
        results_dict['reject_rate1'].append(rj_rate1)
        results_dict['reject_rate2'].append(rj_rate2)
        results_dict['l2_distortion'].append(l2_dist)
    torch.save(results_dict, '{}_results.pth'.format(args.attack))


def pgd_attack(sdim, args):
    thresholds1, thresholds2 = extract_thresholds(sdim, args)
    results_dict = {'reject_rate1': [], 'reject_rate2': [], 'l2_distortion': []}
    eps_list = [0.01, 0.02, 0.05, 0.1]
    for eps in eps_list:
        adversary = LinfPGDAttack(
            sdim, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=-1.0,
            clip_max=1.0, targeted=args.targeted)
        logger.info('epsilon = {:.4f}'.format(adversary.eps))
        #attack_run(sdim, adversary, args)
        l2_dist, rj_rate1, rj_rate2 = adv_eval_with_rejection(sdim, adversary, args, thresholds1, thresholds2)
        results_dict['reject_rate1'].append(rj_rate1)
        results_dict['reject_rate2'].append(rj_rate2)
        results_dict['l2_distortion'].append(l2_dist)
    torch.save(results_dict, '{}_results.pth'.format(args.attack))


def cw_attack(sdim, args):
    thresholds1, thresholds2 = extract_thresholds(sdim, args)
    c_list = [0.1, 1, 10, 100]

    results_dict = {'reject_rate1': [], 'reject_rate2': [], 'l2_distortion': []}
    n_classes = args.get(args.dataset).n_classes
    for c in c_list:
        adversary = CW(sdim, n_classes, max_iterations=1000, c=c, clip_min=0., clip_max=1., learning_rate=0.01, targeted=args.targeted)
        logger.info('coefficient = {:.4f}'.format(c))
        l2_dist, rj_rate1, rj_rate2 = adv_eval_with_rejection(sdim, adversary, args, thresholds1, thresholds2)
        results_dict['reject_rate1'].append(rj_rate1)
        results_dict['reject_rate2'].append(rj_rate2)
        results_dict['l2_distortion'].append(l2_dist)
    torch.save(results_dict, '{}_results.pth'.format(args.attack))


if __name__ == "__main__":
    run()