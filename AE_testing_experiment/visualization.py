from Model import *
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
import numpy as np
import random
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def pca(x, dim=2):
    """
    Perform PCA to for dimensionality reduction

    Arguments:
    ----------
    x: torch.Tensor
        Input data of size (N, K)
    dim: int
        Desired output dimension

    Returns:
    --------
    torch.Tensor
        Output data of size (N, dim)
    """
    n = len(x)
    x_ = x - x.mean(0, keepdim=True)
    gram = x_.matmul(x_.T) / n
    e, v = torch.eig(gram, eigenvectors=True)
    e = e[:, 0]
    eigvec = (x_.T).matmul(v) / torch.sqrt(n * e.unsqueeze(0))

    order = torch.argsort(e, descending=True)
    pc = eigvec[:, order[:dim]]

    return x_.matmul(pc)


def sample(mu, logvar, n):
    mu_ = mu.repeat(n, 1)
    logvar_ = logvar.repeat(n, 1)
    return torch.randn_like(mu_) * torch.exp(0.5 * logvar_) + mu_


def resume(dataset, input_img, z_dim, checkpoint_path, fig_dir, title='', lr=1e-3, device=torch.device("cpu"),
           vae=True):
    if dataset == 'cifar10':
        image_size = 32
        channels = [64, 128, 256]
        ch = 3
    elif dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
    else:
        raise NotImplementedError("dataset is not supported")

    #### Model Optimizer
    model = VAEModel(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size, vae=vae).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #### Resume
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Resume done.')
    model.eval()

    ## input image
    #### Reconstruction
    mu, logvar = model.encoder(input_img)
    return mu, logvar


def visualization(mu, logvar, gamma, fig_dir='./figures'):
    # visualise learned Gaussians
    keep = torch.randperm(len(mu))[:10]
    pts = sample(mu[keep], logvar[keep], n=50)
    pts2d = pca(pts).detach().cpu().numpy()
    plt.figure()
    colours = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcdb22', '#17becf'
    ]
    for i, c in enumerate(colours):
        plt.plot(pts2d[i::10, 0], pts2d[i::10, 1], '.', color=c)
    plt.title('Visualisation of learned Gaussians')
    plt.xlabel('gamma={}'.format(gamma))
    # plt.show()
    plt.savefig(os.path.join(fig_dir, 'gaussia_gamma_{}.png'.format(gamma)))


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    fig_dir = './figures'
    os.makedirs(fig_dir, exist_ok=True)

    # select an image
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(
        CIFAR10(root='/scratch/u7076589/project_VAE/cifar10_ds', train=True, download=True,
                transform=transforms.ToTensor()),
        batch_size=32, shuffle=True, **kwargs)
    # val_loader = DataLoader(
    #     CIFAR10(root='/scratch/u7076589/project_VAE/cifar10_ds', train=False, download=True, transform=transforms.ToTensor()),
    #     batch_size=32, shuffle=True, **kwargs)
    dataiter = iter(train_loader)
    train_img, labels = dataiter.next()  # x.shape #torch.Size([128, 1, 28, 28])

    ## recontruction train
    gamma = 0.01
    checkpoint_path = '/scratch/u7076589/project_VAE/gamma001/cifar10_zdim_128_VAE_gamma_0.01_model.pth'
    # mu logvar
    print(f'Checkpoint{checkpoint_path}')
    mu, logvar = resume('cifar10', train_img, z_dim=128, checkpoint_path=checkpoint_path, fig_dir=fig_dir,
                        title='{}_train'.format(gamma), vae=True)
    visualization(mu, logvar, gamma)
    # ---------------------------------
    gamma = 0.1
    checkpoint_path = '/scratch/u7076589/project_VAE/gamma01/cifar10_zdim_128_VAE_gamma_0.1_model.pth'
    # mu logvar
    print(f'Checkpoint{checkpoint_path}')
    mu, logvar = resume('cifar10', train_img, z_dim=128, checkpoint_path=checkpoint_path, fig_dir=fig_dir,
                        title='{}_train'.format(gamma), vae=True)
    visualization(mu, logvar, gamma)
    # ---------------------------------
    gamma = 1
    checkpoint_path = '/scratch/u7076589/project_VAE/gamma1/cifar10_zdim_128_VAE_gamma_1_model.pth'
    # mu logvar
    print(f'Checkpoint{checkpoint_path}')
    mu, logvar = resume('cifar10', train_img, z_dim=128, checkpoint_path=checkpoint_path, fig_dir=fig_dir,
                        title='{}_train'.format(gamma), vae=True)
    visualization(mu, logvar, gamma)
    # ---------------------------------
    gamma = 10
    checkpoint_path = '/scratch/u7076589/project_VAE/gamma10/cifar10_zdim_128_VAE_gamma_10_model.pth'
    # mu logvar
    print(f'Checkpoint{checkpoint_path}')
    mu, logvar = resume('cifar10', train_img, z_dim=128, checkpoint_path=checkpoint_path, fig_dir=fig_dir,
                        title='{}_train'.format(gamma), vae=True)
    visualization(mu, logvar, gamma)