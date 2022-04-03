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
from os.path import join

def load_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
               crop_height=None, crop_width=None, is_random_crop=True, is_mirror=True, is_gray=False):
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if is_gray == False and img.mode != 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode != 'L':
        img = img.convert('L')

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            # print([w,cropSize])
            cx1 = random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w - crop_width) / 2.))
            cy2 = cy1 = int(round((h - crop_height) / 2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_width, output_height), Image.BICUBIC)
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


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

def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std

if __name__ == '__main__':
    ## device
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # checkpoint_path='/scratch/u7076589/project_VAE/celebA/celeb256_VAE_model.pth'
    checkpoint_path='/scratch/u7076589/project_2stages/celeb256_zdim_256_VAE_gamma_model_overfit.pth'


    model = resume(dataset='celeb256', z_dim=256, checkpoint_path=checkpoint_path, device=device)
    ## transform
    train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
    val_transform = transforms.Compose([
                transforms.ToTensor()
            ])
    data_root = '/scratch/engn8536/Datasets/data256x256'
    image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
    print(f'images: {len(image_list)}')
    image_name = image_list[0]
    print(image_name)
    print(image_name.split('.')[0])
    # img_real = load_image(join(data_root, image_name),input_height=None,
                    # output_height=256)
    #---------------------------------------------------------------------------------
    # training set
    train_size = 20000
    train_list = image_list[:train_size]
    print(f"train_images size: {train_size}")
    val_size = 2000
    print(f"validation_images size: {val_size}")
    val_list = image_list[train_size:train_size+val_size]

    #---------------------------------------------------------------------------------
    # z(mu) training generation
    for img_name in train_list:
        img = load_image(join(data_root,img_name), input_height=None, output_height=256)
        img = train_transform(img)
        # trans_erase_transform(img)  # erase effect
        c,h,w = img.shape
        img = img.view(1,c,h,w).to(device)
        # forward
        mu, logvar, z, y, gamma= model(img)
        torch.save(reparameterize(mu, logvar), '/scratch/u7076589/project_2stages/two_stage_z_train/{}.pt'.format(img_name.split('.')[0]))

    # z(mu) validation generation
    for img_name in val_list:
        img = load_image(join(data_root,img_name), input_height=None, output_height=256)
        img = train_transform(img)
        # trans_erase_transform(img)  # erase effect
        c,h,w = img.shape
        img = img.view(1,c,h,w).to(device)
        # forward
        mu, logvar, z, y, gamma= model(img)
        torch.save(reparameterize(mu, logvar), '/scratch/u7076589/project_2stages/two_stage_z_val/{}.pt'.format(img_name.split('.')[0]))