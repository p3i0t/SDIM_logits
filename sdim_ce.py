import torch
import torch.nn as nn
import math

from losses.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss
from mi_networks import MI1x1ConvNet
from utils import cal_parameters


class ClassConditionalGaussianMixture(nn.Module):
    def __init__(self, n_classes, embed_size):
        super().__init__()
        self.n_classes = n_classes
        self.embed_size = embed_size
        self.class_embed = nn.Embedding(n_classes, embed_size * 2)

    def log_lik(self, x, mean, log_sigma):
        tmp = math.log(2 * math.pi) + 2 * log_sigma + (x - mean).pow(2) * torch.exp(-2 * log_sigma)
        ll = -0.5 * tmp
        return ll

    def forward(self, x):
        # create all class labels for each sample x
        y_full = torch.arange(self.n_classes).unsqueeze(dim=0).repeat(x.size(0), 1).view(-1).to(x.device)

        # repeat each sample for n_classes times
        x = x.unsqueeze(dim=1).repeat(1, self.n_classes, 1).view(x.size(0) * self.n_classes, -1)

        mean, log_sigma = torch.split(self.class_embed(y_full), split_size_or_sections=self.embed_size, dim=-1)

        # evaluate log-likelihoods for each possible (x, y) pairs
        ll = self.log_lik(x, mean, log_sigma).sum(dim=-1).view(-1, self.n_classes)
        return ll


def compute_dim_loss(l_enc, m_enc, measure, mode):
    '''Computes DIM loss.
    Args:
        l_enc: Local feature map encoding.
        m_enc: Multiple globals feature map encoding.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''

    if mode == 'fd':
        loss = fenchel_dual_loss(l_enc, m_enc, measure=measure)
    elif mode == 'nce':
        loss = infonce_loss(l_enc, m_enc)
    elif mode == 'dv':
        loss = donsker_varadhan_loss(l_enc, m_enc)
    else:
        raise NotImplementedError(mode)

    return loss


class FeatureTransformer(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(in_size, 2 * out_size),
                               nn.BatchNorm1d(2 * out_size),
                               nn.ReLU(),
                               # nn.Linear(2 * in_size, 2 * in_size),
                               # nn.BatchNorm1d(2 * in_size),
                               # nn.ReLU(),
                               nn.Linear(2 * out_size, out_size))

    def forward(self, x):
        return self.f(x)


class SDIM(torch.nn.Module):
    def __init__(self, disc_classifier, rep_size=64, n_classes=10, mi_units=64, temperature=10):
        super().__init__()
        self.disc_classifier = disc_classifier #.half()  # Use half-precision for saving memory and time.
        self.disc_classifier.requires_grad_(requires_grad=False)  # shut down grad on pre-trained classifier.
        self.disc_classifier.eval()  # set to eval mode.

        self.rep_size = rep_size
        self.n_classes = n_classes
        self.mi_units = mi_units
        self.temperature = temperature

        self.feature_transformer = FeatureTransformer(self.n_classes, self.rep_size)

        # 1x1 conv performed on only channel dimension
        self.local_MInet = MI1x1ConvNet(self.n_classes, self.mi_units)
        self.global_MInet = MI1x1ConvNet(self.rep_size, self.mi_units)

        self.class_conditional = ClassConditionalGaussianMixture(self.n_classes, self.rep_size)

        self.cross_entropy = nn.CrossEntropyLoss()

        print('==>  # Model parameters.')
        print('==>  # FeatureTransformer parameters: {}.'.format(cal_parameters(self.feature_transformer)))
        print('==>  # T parameters: {}.'.format(cal_parameters(self.local_MInet) + cal_parameters(self.global_MInet)))
        print('==>  # class conditional parameters: {}.'.format(cal_parameters(self.class_conditional)))

    def _T(self, L, G):
        # All globals are reshaped as 1x1 feature maps.
        global_size = G.size()[1:]
        if len(global_size) == 1:
            L = L[:, :, None, None]
            G = G[:, :, None, None]

        L = self.local_MInet(L)
        G = self.global_MInet(G)

        N, local_units = L.size()[:2]
        L = L.view(N, local_units, -1)
        G = G.view(N, local_units, -1)
        return L, G

    def eval_losses(self, x, y, measure='JSD', mode='fd'):
        with torch.no_grad():
            logits = self.disc_classifier(x)  # .half()).float()

        rep = self.feature_transformer(logits)

        # compute mutual infomation loss
        L, G = self._T(logits, rep)
        mi_loss = compute_dim_loss(L, G, measure, mode)

        # evaluate log-likelihoods as logits
        ll = self.class_conditional(rep) / self.rep_size

        # compute cross entropy between of logits and targets
        ce_loss = self.cross_entropy(ll / self.temperature, y)

        # total loss
        loss = mi_loss + ce_loss
        return loss, mi_loss, ce_loss

    def forward(self, x):
        with torch.no_grad():
            logits = self.disc_classifier(x)  # .half()).float()
        rep = self.feature_transformer(logits)
        log_lik = self.class_conditional(rep)
        return log_lik


