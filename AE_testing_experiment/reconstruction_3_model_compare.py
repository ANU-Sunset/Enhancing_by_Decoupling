# reconstruction results between AE and VAE
# 1. reconstruction images
# 2. MSE loss results
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
# Model
from Model import *
from Encoder_Decoder_Model import *
import S1_Model
from S2_Model import *

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
        # print(join(self.root_path, self.image_filenames[index]))
        img = self.input_transform(img)
        return img

    def __len__(self):
        return len(self.image_filenames)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def data(batch_size=8, num_workers=4):
    # train_size = 162770
    train_size = 20000
    # train_size = 1000
    # data_root = 'data256x256'
    data_root = '/scratch/engn8536/Datasets/data256x256'
    image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
    print(f'images: {len(image_list)}')
    train_list = image_list[:train_size]
    # train_list = train_list[:train_size]    # TODO small test
    print(f"train_images size: {len(train_list)}")
    assert len(train_list) > 0
    train_set = ImageDatasetFromFile(train_list, data_root)
    ## validation
    val_size = 2000
    print(f"validation_images size: {val_size}")
    val_list = image_list[train_size:train_size + val_size]
    val_set = ImageDatasetFromFile(val_list, data_root)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    print(len(val_loader))
    return train_loader, val_loader


def calc_reconstruction_loss(x, recon_x):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
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


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------
    ## device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)

    # ---------------------------------------------------------------------------------
    ## build model
    channels = [64, 128, 256, 512, 512, 512]
    image_size = 256
    ch = 3
    zdim = 256
    model_AE = SoftIntroVAE(cdim=ch, zdim=zdim, channels=channels, image_size=image_size).to(device)
    # print(model.encoder)
    optimizer_AE = optim.Adam(model_AE.parameters(), lr=2e-4)

    model_VAE = VAEModel(cdim=ch, zdim=zdim, channels=channels, image_size=image_size, vae=True).to(device)
    optimizer_VAE = optim.Adam(model_VAE.parameters(), lr=2e-4)

    model_s1 = S1_Model.VAEModel(cdim=ch, zdim=zdim, channels=channels, image_size=image_size, vae=True).to(device)
    optimizer_s1 = optim.Adam(model_s1.parameters(), lr=2e-4)

    model_s2 = S2Model(dim=256, second_dim=1024).to(device)
    optimizer_s2 = optim.Adam(model_s2.parameters(), lr=2e-4)
    # ---------------------------------------------------------------------------------
    # model paths
    ae_checkpoint_path = '/scratch/u7076589/project_DA_loss/celeb256_zdim_256_DA_lossmodel.pth'

    vae_checkpoint_path = '/scratch/u7076589/project_VAE/celebA/celeb256_zdim_256_VAE_gamma_0.5_model.pth'

    # load models /Resume
    ## AE
    checkpoint = torch.load(ae_checkpoint_path)
    model_AE.load_state_dict(checkpoint['model'])
    optimizer_AE.load_state_dict(checkpoint['optimizer'])
    print('Resume AE done.')
    model_AE.eval()

    ## VAE
    checkpoint = torch.load(vae_checkpoint_path)
    model_VAE.load_state_dict(checkpoint['model'])
    optimizer_VAE.load_state_dict(checkpoint['optimizer'])
    print('Resume VAE done.')
    model_VAE.eval()

    ## 2 stage VAE
    checkpoint_path_1st = '/scratch/u7076589/project_2stages/celeb256_zdim_256_VAE_gamma_model.pth'  # TODO: 1st path
    print(f'Checkpoint 1st: {checkpoint_path_1st}')

    checkpoint_path_2nd = '/scratch/u7076589/project_2stages/stage2/2nd_stage_loss_plot/celeb256_2nd_stage__VAE_gamma_model.pth'  # TODO: 2nd path
    print(f'Checkpoint 2nd: {checkpoint_path_2nd}')

    # 1st stage
    checkpoint = torch.load(checkpoint_path_1st)
    model_s1.load_state_dict(checkpoint['model'])
    optimizer_s1.load_state_dict(checkpoint['optimizer'])
    gamma = checkpoint['gamma']
    print(gamma.data.cpu().item)

    # 2nd stage
    checkpoint = torch.load(checkpoint_path_2nd)
    model_s2.load_state_dict(checkpoint['model'])
    model_s2.load_state_dict(checkpoint['model'])
    optimizer_s2.load_state_dict(checkpoint['optimizer'])
    gamma_s2 = checkpoint['gamma']
    print(gamma_s2.data.cpu().item)

    # ---------------------------------------------------------------------------------
    ## input images
    batch_size = 10
    train_loader, val_loader = data(batch_size=batch_size)
    # dataiter = iter(val_loader)

    # ---------------------------------------------------------------------------------
    ## Reconstruction
    # AE
    fig_dir = '/scratch/u7076589/project_recon_compare/recon_img'
    # x = dataiter.next()
    # x = x.to(device)
    # z,y_AE=model_AE(x)
    # print(y_AE.shape)

    # VAE
    # mu, _= model_VAE.encoder(x)
    # y_VAE = model_VAE.decoder(mu)
    print('----------------------------------------------------------')
    # ---------------------------------------------------------------------------------
    # train
    # results_loss = {'AE':0, 'VAE':0,'2_stage':0}
    # with torch.no_grad():
    #     for x in train_loader:
    #         # print(batch.shape)
    #         # dataiter = iter(val_loader)
    #         # x = dataiter.next()
    #         x=x.to(device)
    #         # print('x{}'.format(x.shape))
    #         # #AE
    #         z,y_AE=model_AE(x)
    #         loss_AE = calc_reconstruction_loss(x, y_AE)
    #         results_loss['AE']+=loss_AE.data.cpu().item()/len(train_loader)

    #         # # VAE
    #         mu, _= model_VAE.encoder(x)
    #         y_VAE = model_VAE.decoder(mu)
    #         loss_VAE = calc_reconstruction_loss(x, y_VAE)
    #         results_loss['VAE']+=loss_VAE.data.cpu().item()/len(train_loader)

    #         # # two stage VAE
    #         z, _ = model_s1.encoder(x)
    #         mu, _ = model_s2.encoder(z)
    #         y_2VAE = model_s2.decoder(mu)
    #         y_2VAE = model_s1.decoder(y_2VAE)
    #         loss_2VAE = calc_reconstruction_loss(x,y_2VAE)
    #         results_loss['2_stage']+=loss_2VAE.data.cpu().item()/len(train_loader)

    print('AE train MSE loss: {}'.format(results_loss['AE']))
    print('VAE train MSE loss: {}'.format(results_loss['VAE']))
    print('2-VAE train MSE loss: {}'.format(results_loss['2_stage']))
    max_imgs = min(batch_size, 10)

    vutils.save_image(
        torch.cat([x[:max_imgs], y_AE[:max_imgs], y_VAE[:max_imgs], y_2VAE[:max_imgs]], dim=0).data.cpu(),
        '{}/train_rec_image.jpg'.format(fig_dir), nrow=10)
    print('----------------------------------------------------------')
    # validation
    results_loss = {'AE': 0, 'VAE': 0, '2_stage': 0}
    with torch.no_grad():
        for x in val_loader:
            # print(batch.shape)
            # dataiter = iter(val_loader)
            # x = dataiter.next()
            x = x.to(device)
            # print('x{}'.format(x.shape))
            # #AE
            z, y_AE = model_AE(x)
            loss_AE = calc_reconstruction_loss(x, y_AE)
            results_loss['AE'] += loss_AE.data.cpu().item() / len(val_loader)

            # # VAE
            mu, _ = model_VAE.encoder(x)
            y_VAE = model_VAE.decoder(mu)
            loss_VAE = calc_reconstruction_loss(x, y_VAE)
            results_loss['VAE'] += loss_VAE.data.cpu().item() / len(val_loader)

            # # two stage VAE
            z, _ = model_s1.encoder(x)
            mu, _ = model_s2.encoder(z)
            y_2VAE = model_s2.decoder(mu)
            y_2VAE = model_s1.decoder(y_2VAE)
            loss_2VAE = calc_reconstruction_loss(x, y_2VAE)
            results_loss['2_stage'] += loss_2VAE.data.cpu().item() / len(val_loader)

    print('AE validation MSE loss: {}'.format(results_loss['AE']))
    print('VAE validation MSE loss: {}'.format(results_loss['VAE']))
    print('2-VAE validation MSE loss: {}'.format(results_loss['2_stage']))
    max_imgs = min(batch_size, 10)

    vutils.save_image(
        torch.cat([x[:max_imgs], y_AE[:max_imgs], y_VAE[:max_imgs], y_2VAE[:max_imgs]], dim=0).data.cpu(),
        '{}/val_rec_image.jpg'.format(fig_dir), nrow=10)


