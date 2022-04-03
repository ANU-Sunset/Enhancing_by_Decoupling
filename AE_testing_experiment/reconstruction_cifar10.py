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

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def calc_reconstruction_loss(x, recon_x):
    """
    :param x: original inputs
    :param recon_x:  AE_testing_experiment of the VAE's input
    :param loss_type: "mse",
    :param reduction: "mean"
    :return: recon_loss
    """
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    recon_error = F.mse_loss(recon_x, x, reduction='none')
    recon_error = recon_error.sum(1)
    recon_error = recon_error.mean()

    return recon_error

def resume(dataset, input_img, z_dim, checkpoint_path, fig_dir, title='',lr=1e-3, device=torch.device("cpu"),vae=True):
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
    model = VAEModel(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size,vae=vae).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #### Resume
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Resume done.')
    model.eval()

    ## input image
    #### Reconstruction
    z, logvar = model.encoder(input_img)
    y = model.decoder(z)

    max_imgs = min(input_img.size(0), 4)
    vutils.save_image(
        torch.cat([input_img[:max_imgs], y[:max_imgs]], dim=0).data.cpu(),
        '{}/{}_rec_image.jpg'.format(fig_dir,title), nrow=4)
    # MSE = torch.nn.MSELoss()
    loss = calc_reconstruction_loss(input_img,y)
    print(loss)


if __name__ == '__main__':
    ## device
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # save AE_testing_experiment path
    fig_dir = '/scratch/u7076589/project_Autoencoder/gamma_reconstruction'  # TODO: path
    # fig_dir = '/scratch/u7076589/project_VAE/gamma_reconstruction'
    ## checkpoint path
    gamma =0.1 # TODO: gamma
    checkpoint_path = '/scratch/u7076589/project_Autoencoder/gamma01/cifar10_zdim_128_VAE_gamma_0.1_model.pth'  # TODO: path
    print(f'Checkpoint{checkpoint_path}')

    # select an image
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(
        CIFAR10(root='./cifar10_ds', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=32, shuffle=True, **kwargs)
    val_loader = DataLoader(
        CIFAR10(root='./cifar10_ds', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=32, shuffle=True, **kwargs)

    ## recontruction train
    dataiter = iter(train_loader)
    train_img, labels = dataiter.next()  # x.shape #torch.Size([128, 1, 28, 28])
    resume('cifar10', train_img, z_dim=128, checkpoint_path=checkpoint_path, fig_dir=fig_dir,title='{}_train'.format(gamma),vae=False)

    ## AE_testing_experiment validation
    dataiter = iter(val_loader)
    val_img, labels = dataiter.next()  # x.shape #torch.Size([128, 1, 28, 28])
    resume('cifar10', val_img, z_dim=128, checkpoint_path=checkpoint_path, fig_dir=fig_dir, title='{}_val'.format(gamma),vae=False)