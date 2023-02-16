import torch
import torch.nn as nn
from torch import optim

def get_optim(parameters, lr, config):
    return optim.Adam(parameters, lr, [config.beta1, config.beta2])

def get_d(config):
    net = DUAL(config.d_n_layers, config.activation)
    net.apply(weights_init_d)
    if torch.cuda.is_available():
        net.cuda()
    return net

def get_g(config):
    residual = False  # (config.solver != 'bary_ot')
    net = GEN(config.g_n_layers, config.g_norm, config.activation, residual)
    if residual:
        net.apply(weights_init_g)
    if torch.cuda.is_available():
        net.cuda()
    return net

class DUAL(nn.Module):
    def __init__(self, n_layers, activation):
        super(DUAL, self).__init__()
        in_h = 32*32  # 16*16
        out_h = 200
        modules = []
        for i in range(n_layers):
            modules.append(nn.Linear(in_h, out_h))
            modules.append(get_activation(activation))
            in_h = out_h
            out_h = 1000  # 500
        modules.append(nn.Linear(1000, 1))  # modules.append(nn.Linear(500, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.model(input)
        return output.squeeze()

class GEN(nn.Module):
    def __init__(self, n_layers, norm, activation, residual):
        super(GEN, self).__init__()
        self.residual = False  # self.residual = residual
        in_h = 100  # 16*16
        out_h = 200
        modules = []
        for i in range(n_layers):
            m = nn.Linear(in_h, out_h)
            norm_ms = apply_normalization(norm, out_h, m)
            for nm in norm_ms:
                modules.append(nm)
            modules.append(get_activation(activation))
            in_h = out_h
            out_h = 1000  # 500
        modules.append(nn.Linear(1000, 32 * 32))  # modules.append(nn.Linear(500, 16*16))
        modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, input, template):
        output = self.model(input.view(input.size(0), -1))
        output = output.view(*template.size())
        return 2*output + input if self.residual else output

def get_activation(ac):
    if ac == 'relu':
        return nn.ReLU()
    elif ac == 'elu':
        return nn.ELU()
    elif ac == 'leakyrelu':
        return nn.LeakyReLU(0.2)
    elif ac == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError('activation [%s] is not found' % ac)

def apply_normalization(norm, dim, module):
    """
    Applies normalization `norm` to `module`.
    Optionally uses `dim`
    Returns a list of modules.
    """
    if norm == 'none':
        return [module]
    elif norm == 'batch':
        return [module, nn.BatchNorm1d(dim)]
    elif norm == 'layer':
        return [module, nn.GroupNorm(1, dim)]
    elif norm == 'spectral':
        return [torch.nn.utils.spectral_norm(module, name='weight')]
    else:
        raise NotImplementedError('normalization [%s] is not found' % norm)

def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
