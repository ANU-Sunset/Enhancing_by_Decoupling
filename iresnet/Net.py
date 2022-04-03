import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn import Parameter


class ActNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(ActNorm, self).__init__()
        self.eps = eps
        self._log_scale = Parameter(torch.Tensor(dim))
        self._shift = Parameter(torch.Tensor(dim))
        self._init = False

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                mean = x.mean(dim=0)
                zero_mean = x - mean
                var = (zero_mean ** 2).mean(dim=0)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self._log_scale
        logdet = log_scale.sum()
        return x * torch.exp(log_scale) + self._shift, logdet

    def inverse(self, x):
        return (x - self._shift) * torch.exp(-self._log_scale)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block_num = 100
        self.block_list = nn.ModuleList()
        for i in range(self.block_num):
            self.block_list.append(res_block())

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x

    def inverse(self, y, max_iter=100):
        with torch.no_grad():
            for block in reversed(self.block_list):
                y = block.inverse(y, maxIter=max_iter)
            return y


class res_block(nn.Module):
    def __init__(self, D=256, coeff=.87):
        super(res_block, self).__init__()
        self.coeff = coeff
        layers = []
        layers.append(nn.ELU())
        layers.append(spectral_norm(nn.Linear(D, D), self.coeff, 100))
        # layers.append(nn.Linear(D, D))
        layers.append(nn.ELU())
        layers.append(spectral_norm(nn.Linear(D, D), self.coeff, 100))
        # layers.append(nn.Linear(D, D))
        layers.append(nn.ELU())
        layers.append(spectral_norm(nn.Linear(D, D), self.coeff, 100))
        # layers.append(nn.Linear(D, D))
        self.basic_block = nn.Sequential(*layers)
        self.actnorm = ActNorm(D)

    def forward(self, x):
        x = self.actnorm(x)[0]
        Fx = self.basic_block(x)
        y = Fx + x
        return y

    def inverse(self, y, maxIter=100):
        # algorithm
        x = y
        for _ in range(maxIter):
            gxi = self.basic_block(x)
            x = y - gxi
        x = self.actnorm.inverse(x)
        return x


def spectral_norm(module, coeff, iters):
    module.weight.data = module.weight.data * 1000
    W = module.weight.data
    h, w = W.size()
    u = normalize(W.new_empty(h).normal_(0, 1), dim=0, eps=1e-12)
    v = normalize(W.new_empty(w).normal_(0, 1), dim=0, eps=1e-12)

    with torch.no_grad():
        for _ in range(iters):
            v = normalize(torch.mv(W.t(), u), dim=0, eps=1e-12, out=v)
            u = normalize(torch.mv(W, v), dim=0, eps=1e-12, out=u)

    sigma = torch.dot(u, torch.mv(W, v))

    factor = max(1.0, (sigma / coeff).item())
    weight = W / (factor + 1e-5)
    module.weight.data = weight
    return module

"""
diff = lambda x, y: torch.norm((x - y), p=2) / torch.norm(x, p=2)
x = torch.rand((100, 2), requires_grad=True)
net = Net()
y = net(x)
x1 = net.inverse(y, 50)
print(diff(x, x1))
"""





