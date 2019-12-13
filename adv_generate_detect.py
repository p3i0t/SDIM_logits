"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import models
import os

from sklearn.linear_model import LogisticRegressionCV

from torch.utils.data import TensorDataset, DataLoader
from sdim_ce import SDIM


def load_pretrained_model(args):
    checkpoint_path = '{}_{}.pth'.format(args.classifier_name, args.problem)
    print('Load pre-trained checkpoint: {}'.format(checkpoint_path))
    pre_trained_dir = os.path.join(args.log_dir, checkpoint_path)

    model = models.ResNet34(num_c=args.n_classes)
    model.load_state_dict(torch.load(pre_trained_dir, map_location=lambda storage, loc: storage))
    return model


def generate_score(sdim, args):
    # load dataset
    print('load target data: ', args.problem)
    outf = args.outf + '{}_{}/'.format(args.classifier_name, args.problem)
    test_clean_data = torch.load(outf + 'clean_data_%s_%s_%s.pth' % (args.classifier_name, args.problem, args.adv_type))
    test_adv_data = torch.load(outf + 'adv_data_%s_%s_%s.pth' % (args.classifier_name, args.problem, args.adv_type))
    test_noisy_data = torch.load(outf + 'noisy_data_%s_%s_%s.pth' % (args.classifier_name, args.problem, args.adv_type))
    test_label = torch.load(outf + 'label_%s_%s_%s.pth' % (args.classifier_name, args.problem, args.adv_type))

    sdim.eval()

    print(test_clean_data.size())
    clean_loader = DataLoader(TensorDataset(test_clean_data), batch_size=args.batch_size)
    adv_loader = DataLoader(TensorDataset(test_adv_data), batch_size=args.batch_size)

    adv_score_list, clean_score_list = [], []
    # get clean score
    for idx in range(test_clean_data.size(0) // args.batch_size):
        #print(len(x), x)
        x = test_clean_data[idx * args.batch_size: (idx + 1) * args.batch_size].to(args.device)
        logits = sdim(x)  # class conditionals as losits
        clean_score_list.append(logits.cpu())

    clean_score = torch.cat(clean_score_list, dim=0)
    clean_y = torch.zeros(clean_score.size(0), 1)

    for idx in range(test_adv_data.size(0) // args.batch_size):
        x = test_adv_data[idx * args.batch_size: (idx + 1) * args.batch_size].to(args.device)
        logits = sdim(x)  # class conditionals as losits
        adv_score_list.append(logits.cpu())

    adv_score = torch.cat(adv_score_list, dim=0)
    adv_y = torch.zeros(adv_score.size(0), 1)

    x = torch.cat([clean_score, adv_score], dim=0)
    y = torch.cat([clean_y, adv_y], dim=0)
    save_path = os.path.join(outf, 'score_%s_%s_%s.pth' % (args.classifier_name, args.problem, args.adv_type))
    print('data size: ', x.size(0))
    torch.save({'x': x, 'y': y}, save_path)


def detect(args):
    # initial setup
    dataset_list = ['cifar10', 'svhn']
    adv_test_list = ['FGSM', 'BIM', 'DeepFool', 'CWL2']

    from sklearn.metrics import roc_auc_score

    def split_set(x, y, ratio=0.1):
        n = x.size(0)
        n_select = int(n * ratio)
        indices = torch.randperm(n).tolist()
        x1, x2 = x[indices[:n_select]], x[indices[n_select:]]
        y1, y2 = y[indices[:n_select]], y[indices[n_select:]]
        return x1, y1, x2, y2

    for dataset in dataset_list:
        print('load train data: ', dataset)
        outf = './adv_output/' + args.classifier_name + '_' + dataset + '/'

        list_best_results_out, list_best_results_index_out = [], []
        for adv in adv_test_list:
            save_path = os.path.join(outf, 'score_%s_%s_%s.pth' % (args.classifier_name, args.problem, adv))
            data = torch.load(save_path)

            # Split follows https://github.com/wangxin0716/SDIM_logits_private/blob/master/lib_regression.py
            X_val, Y_val, X_test, Y_test = split_set(data['x'], data['y'], ratio=0.1)
            X_train, Y_train, X_val_for_test, Y_val_for_test = split_set(X_val, Y_val, ratio=0.5)
            lr = LogisticRegressionCV(n_jobs=-1).fit(X_train.detach().numpy(), Y_train.detach().numpy())
            # y_pred = lr.predict_proba(X_train)[:, 1]
            #
            # y_pred = lr.predict_proba(X_val_for_test)[:, 1]
            y_pred = lr.predict_proba(X_test.numpy())[:, 1]
            auroc = roc_auc_score(Y_test.numpy(), y_pred)
            print('{}, {}, auroc: {:.4f}'.format(dataset, adv, auroc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of SDIM_logits.')
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--extract", action="store_true",
                        help="Used in inference mode")
    parser.add_argument("--detect", action="store_true",
                        help="Used in inference mode with rejection")
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
    parser.add_argument('--batch_size', type=int, default=200, metavar='N', help='batch size for data loader')
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
                        default=64, help="output size of 1x1 conv network for mutual information estimation")
    parser.add_argument("--rep_size", type=int,
                        default=10, help="size of the global representation from encoder")
    parser.add_argument("--classifier_name", type=str, default='resnet',
                        help="classifier name: resnet | densenet")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Ablation
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument('--adv_type', required=True, help='FGSM | BIM | DeepFool | CWL2')
    parser.add_argument('--outf', default='./adv_output/', help='folder to output results')

    args = parser.parse_args()  # So error if typo

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if use_cuda else "cpu")

    classifier = load_pretrained_model(args).to(args.device)
    sdim = SDIM(disc_classifier=classifier,
                n_classes=args.n_classes,
                rep_size=args.rep_size,
                mi_units=args.mi_units,
                ).to(args.device)

    if args.extract:
        generate_score(sdim, args)
    elif args.detect:
        detect(args)
