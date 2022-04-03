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

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


class ImageDatasetFromFile(data.Dataset):
    def __init__(self, image_list, root_path):
        super(ImageDatasetFromFile, self).__init__()
        self.root_path = root_path
        self.image_filenames = image_list
        self.input_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(join(self.root_path, self.image_filenames[index]))
        img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def resume(dataset, z_dim, checkpoint_path, device=torch.device("cpu"), lr=1e-3, vae=True):
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
    return model


def change_z_dim(z, changed_dim=0):
    one = torch.zeros([6, 256]).to(device)
    one[0] = z
    one[1] = z
    one[2] = z
    one[3] = z
    one[4] = z
    one[5] = z
    one[1, changed_dim] = one[1, changed_dim] * -1
    one[2, changed_dim] = one[2, changed_dim] * -2
    one[3, changed_dim] = one[3, changed_dim] * -3
    one[4, changed_dim] = one[4, changed_dim] * -4
    one[5, changed_dim] = one[5, changed_dim] * -6

    return one


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 20
    # TODO small test
    # train_size = 162770
    train_size = 20
    # train_size = 1000
    data_root = '/scratch/u7076589/project_VAE/celebA_3'
    # data_root = '/scratch/engn8536/Datasets/data256x256'
    image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
    print(f'images: {len(image_list)}')
    train_list = image_list[:train_size]
    print(f"train_images size: {train_size}")
    assert len(train_list) > 0
    train_set = ImageDatasetFromFile(train_list, data_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    checkpoint_path = '/scratch/u7076589/project_VAE/celebA/celeb256_zdim_256_VAE_gamma_0.5_model.pth'
    #### Model Optimizer
    model = resume('celeb256', 256, checkpoint_path=checkpoint_path, device=device)
    model.eval()

    fig_dir = '/scratch/u7076589/project_VAE/INT_zdim'
    dataiter = iter(train_loader)
    x = dataiter.next()
    x = x.to(device)
    mu, _ = model.encoder(x)
    print(mu.shape)  # 20x256

    # test one image
    line = None
    for i in range(256):
        z = change_z_dim(mu[10].to(device), changed_dim=i)
        # print(z.shape)
        rec_img = model.decoder(z[0].view(1, 256))
        rec_img_0 = model.decoder(z[1].view(1, 256))
        rec_img_1 = model.decoder(z[2].view(1, 256))
        rec_img_2 = model.decoder(z[3].view(1, 256))
        rec_img_3 = model.decoder(z[4].view(1, 256))
        rec_img_4 = model.decoder(z[5].view(1, 256))
        one = torch.cat([rec_img, rec_img_0, rec_img_1, rec_img_2, rec_img_3, rec_img_4], dim=0)
        if line == None:
            line = one
        else:
            line = torch.cat((line, one), 0)
        # print(line.size())
        if (i + 1) % 8 == 0:
            vutils.save_image(
                line,
                '{}/rec_image_{}.jpg'.format('./INT', i), nrow=6)
            line = None