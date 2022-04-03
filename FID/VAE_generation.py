from Model import *
import torch.utils.data as data
from os.path import join
from PIL import Image
import torchvision.transforms as transforms

import torch
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from time import perf_counter
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from copy import deepcopy

def resume(dataset, z_dim, checkpoint_path, device=torch.device("cpu"),lr=1e-3,vae=True):
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
    return model

def sample(model, num_samples, latent_dim, current_device, idx, fig_dir='./'):
    """
    Samples from the latent space and return the corresponding
    image space map.
    :param num_samples: (Int) Number of samples
    :param current_device: (Int) Device to run the model
    :return: (Tensor)
    """
    z = torch.randn(num_samples,latent_dim)

    z = z.to(current_device)

    y = model.decoder(z)
    max_imgs = min(num_samples, 4)
    vutils.save_image(
        torch.cat([y[:max_imgs]], dim=0).data.cpu(),
        '{}/gen_image_{}.jpg'.format(fig_dir,idx), nrow=4)
    # # return samples

if __name__ == '__main__':
    ## device
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    checkpoint_path='/scratch/u7076589/project_VAE/celebA/celeb256_VAE_model.pth'
    model = resume(dataset='celeb256', z_dim=256, checkpoint_path=checkpoint_path, device=device)
    num_samples=1 # FID score
    for i in range(200):
        sample(model, num_samples=num_samples, latent_dim=256, current_device=device, idx=i,fig_dir='/scratch/u7076589/project_VAE/vae_celebA_gen')