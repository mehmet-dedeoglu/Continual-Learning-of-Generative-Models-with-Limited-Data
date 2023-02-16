import torch.nn as nn
import math
import torch.nn.functional as F
import torch


def _scaleFunc_pos(mu, std, th_pos):
    alpha = (th_pos - mu) / std
    pdf = torch.exp(-(alpha.pow(2) / 2)) / math.sqrt(2 * math.pi)
    cdf = (1 + torch.erf(alpha / math.sqrt(2))) / 2

    return mu + std * (pdf / (1 - cdf))


def _scaleFunc_neg(mu, std, th_neg):
    beta = (th_neg - mu) / std
    pdf = torch.exp(-(beta.pow(2) / 2)) / math.sqrt(2 * math.pi)
    cdf = (1 + torch.erf(beta / math.sqrt(2))) / 2

    return mu - std * (pdf / cdf)


class _quanFunc(torch.autograd.Function):
    def __init__(self, mu, delta, alpha):
        super(_quanFunc, self).__init__()
        self.mu = mu.item()
        self.delta = delta.item()
        self.alpha = alpha.item()

    def forward(self, input):
        output = input.clone().zero_()
        output[input.gt(self.mu + self.delta)] = 1
        output[input.lt(self.mu - self.delta)] = -1

        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone() / self.alpha
        return grad_input


class quanConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, n_thresholds=1):
        super(quanConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        self.mu, self.std = self.weight.mean(), self.weight.std()

        max_w_init = self.weight.abs().max()
        self.init_factor = torch.Tensor([0.05])
        deltas = (max_w_init) * self.init_factor

        self.delta_th = nn.Parameter(deltas)
        self.th_clip = F.hardtanh(self.delta_th.abs(), min_val=(-3 * self.std).item(), max_val=(3 * self.std).item())
        self.scale_factor = _scaleFunc_pos(self.mu, self.std, self.mu + self.th_clip)

    def forward(self, input):
        self.mu = self.weight.mean()
        self.std = self.weight.std()
        self.th_clip = F.hardtanh(self.delta_th.abs(), min_val=(-3 * self.std).item(), max_val=(3 * self.std).item())
        self.scale_factor = _scaleFunc_pos(self.mu, self.std, self.mu + self.th_clip)

        for idx, param in enumerate(self.th_clip):
            tern_weight = _quanFunc(self.mu, param, self.scale_factor[idx])(self.weight) * self.scale_factor[idx]
            if idx == 0:
                output = F.conv2d(input, tern_weight, self.bias, self.stride, self.padding, self.dilation,
                                  self.groups)
            else:
                output += F.conv2d(input, tern_weight, self.bias, self.stride, self.padding, self.dilation,
                                   self.groups)

        return output


class quanConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True, n_thresholds=1):
        super(quanConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                                  padding, output_padding, groups, bias, dilation)
        self.mu, self.std = self.weight.mean(), self.weight.std()

        max_w_init = self.weight.abs().max()
        self.init_factor = torch.Tensor([0.05])
        deltas = (max_w_init) * self.init_factor

        self.delta_th = nn.Parameter(deltas)
        self.th_clip = F.hardtanh(self.delta_th.abs(), min_val=(-3 * self.std).item(), max_val=(3 * self.std).item())
        self.scale_factor = _scaleFunc_pos(self.mu, self.std, self.mu + self.th_clip)

    def forward(self, input, output_size=None):
        self.mu = self.weight.mean()
        self.std = self.weight.std()
        self.th_clip = F.hardtanh(self.delta_th.abs(), min_val=(-3 * self.std).item(), max_val=(3 * self.std).item())
        self.scale_factor = _scaleFunc_pos(self.mu, self.std, self.mu + self.th_clip)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        for idx, param in enumerate(self.th_clip):
            tern_weight = _quanFunc(self.mu, param, self.scale_factor[idx])(self.weight) * self.scale_factor[idx]
            if idx == 0:
                output = F.conv_transpose2d(input, tern_weight, self.bias, self.stride, self.padding,
                                            output_padding, self.groups, self.dilation)
            else:
                output += F.conv_transpose2d(input, tern_weight, self.bias, self.stride, self.padding,
                                             output_padding, self.groups, self.dilation)

        return output
