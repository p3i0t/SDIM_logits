import torch
from torch.optim import Adam


def _replicate_input(x):
    return x.detach().clone()


def _arctanh(x):
    return (torch.log((1 + x) / (1 - x))) * 0.5


def _to_one_hot(y, num_classes=10):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    y = _replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot


def _scale(x, clip_min=0., clip_max=1.):
    """Scale to [-1, 1].  """
    #x_shift = x
    x_shift = (x - clip_min) / (clip_max - clip_min)
    return x_shift * 2 - 1


def _inv_scale(x, clip_min, clip_max):
    """Scale from [-1, 1] to [clip_min, clip_max].  """
    x_shift = (x + 1) / 2
    x_shift = x_shift * (clip_max - clip_min) + clip_min
    return x_shift


class CW:
    def __init__(self, predict, n_classes, c=1, confidence=0, targeted=False, learning_rate=0.01,
                 max_iterations=1000, clip_min=0., clip_max=1., norm=2):
        self.predict = predict
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.c = c
        self.targeted = targeted
        self.norm = norm
        self.confidence = confidence
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _adv_loss_fn(self, pred, y_onehot):
        real = (pred * y_onehot).sum(dim=1)

        other = (1 - y_onehot) * pred - y_onehot * 1e6
        other = other.max(dim=1)[0]  # get the other max logit values

        if self.targeted:
            loss = torch.relu(other - real + self.confidence)  # real bigger than other at least confidence
        else:
            loss = torch.relu(real - other + self.confidence)  # other bigger than other at least confidence
        return loss

    def _distort_loss_fn(self, adv_x, x, norm):
        diff = (adv_x - x).view(adv_x.size(0), -1)
        if norm == 2:
            loss = torch.norm(diff, p=2, dim=-1)
        elif norm == 'inf':
            loss = torch.abs(diff).max(dim=1)[0]
        return loss

    def perturb(self, x, y):
        x = _replicate_input(x)
        x_scale = _scale(x, self.clip_min, self.clip_max)  # scale to [-1, 1]
        x_scale = _arctanh(x_scale)  # scale to [-inf, inf]
        y_onehot = _to_one_hot(y, self.n_classes).float()

        delta = torch.nn.Parameter(torch.zeros_like(x_scale))
        optimizer = Adam([delta], lr=self.learning_rate)

        for step in range(self.max_iterations):
            optimizer.zero_grad()

            adv = _inv_scale(torch.tanh(x_scale + delta), clip_min=self.clip_min, clip_max=self.clip_max)

            pred = self.predict(adv)
            distort_loss = self._distort_loss_fn(adv, x, self.norm)
            adv_loss = self._adv_loss_fn(pred, y_onehot)
            loss = distort_loss + self.c * adv_loss
            loss.sum().backward()
            optimizer.step()

        return adv.data, distort_loss.data, adv_loss.data, loss.data


