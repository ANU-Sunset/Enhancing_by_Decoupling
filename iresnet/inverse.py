import torch
import os
from Net import Net
from torch.nn.functional import normalize

def spectral_norm(module, coeff, iters):
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
    weight = W / factor
    module.weight.data = weight


os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"


diff = lambda x, y: torch.norm((x - y), p=2) / torch.norm(x, p=2)
resume_checkpoint_path = 'model.pt'
net = torch.load(resume_checkpoint_path).cuda()
x = torch.randn((100, 256), requires_grad=True).cuda()
y = net(x)
x1 = net.inverse(y, 50)
print(diff(x, x1))

