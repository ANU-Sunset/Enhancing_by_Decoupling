# compare reconstruction MSE between our AE and 1st stage VAE of two-stage VAE
from Encoder_Decoder_Model import *
from Model import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
# from helper import *
from time import perf_counter
import os
import random
from dataset import *
from tqdm import tqdm
import torchvision.utils as vutils
import torch.nn.functional as F

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


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


def resume(dataset, input_img, z_dim, checkpoint_path, fig_dir, title='', lr=2e-4, device=torch.device("cpu"),
           vae=True):
    if dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
    else:
        raise NotImplementedError("dataset is not supported")

    #### Model Optimizer
    if vae:
        model = VAEModel(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size, vae=vae).to(device)
    else:
        model = SoftIntroVAE(cdim=ch, channels=channels, image_size=image_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #### Resume
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if vae:
        gamma = checkpoint['gamma']
    print('Resume done.')
    # print(model)
    model.eval()

    ## input image
    #### Reconstruction
    if vae:
        z, logvar = model.encoder(input_img)
        y = model.decoder(z)
    else:
        z, y = model(input_img)

    # print(y.shape)
    max_imgs = min(input_img.size(0), 4)
    vutils.save_image(
        torch.cat([input_img[:max_imgs], y[:max_imgs]], dim=0).data.cpu(),
        '{}/{}_rec_image.jpg'.format(fig_dir, title), nrow=4)
    # vutils.save_image(y,'{}/{}_rec_image.jpg'.format(fig_dir,title), nrow=4)
    # MSE = torch.nn.MSELoss()
    loss = calc_reconstruction_loss(input_img, y)
    return loss


if __name__ == '__main__':
    ## device
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # save reconstruction path
    fig_dir = '/scratch/u7076589/project_2stages/compare_MSE_plot'  # TODO: path

    # pretrained models
    checkpoint_path_AE = '/scratch/u7076589/project_DA_loss/celeb256_zdim_256_DA_lossmodel.pth'  # TODO: path
    print(f'Checkpoint AE: {checkpoint_path_AE}')

    checkpoint_path_1st = '/scratch/u7076589/project_2stages/celeb256_zdim_256_VAE_gamma_model.pth'  # TODO: path
    print(f'Checkpoint 1st: {checkpoint_path_1st}')

    # select an image
    # TODO small test
    # train_size = 162770
    train_size = 20000
    # train_size = 10
    # data_root = 'data256x256'
    data_root = '/scratch/engn8536/Datasets/data256x256'
    image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
    print(f'images: {len(image_list)}')
    train_list = image_list[:train_size]
    print(f"train_images size: {train_size}")
    assert len(train_list) > 0
    train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                     output_height=256, is_mirror=True)
    ## validation
    val_size = 2000
    print(f"validation_images size: {val_size}")
    val_list = image_list[train_size:train_size + val_size]
    val_set = ImageDatasetFromFile(val_list, data_root, input_height=None, crop_height=None,
                                   output_height=256, is_mirror=True, train=False)

    train_data_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                                   num_workers=4)
    val_data_loader = DataLoader(val_set, batch_size=8, shuffle=True,
                                 num_workers=4)

    ## recontruction train
    # 1. our AE
    dataiter = iter(train_data_loader)
    train_img = dataiter.next()  # x.shape #torch.Size([128, 1, 28, 28])
    # print(len(train_img)) #2
    # print(train_img[0].shape) #torch.Size([8, 3, 256, 256])
    loss = resume('celeb256', train_img[0], z_dim=256, checkpoint_path=checkpoint_path_AE, fig_dir=fig_dir,
                  title='{}_train'.format('AE'), vae=False)
    print(f'AE train MSE loss: {loss}')
    # 2.two-stage VAE 1st
    loss = resume('celeb256', train_img[0], z_dim=256, checkpoint_path=checkpoint_path_1st, fig_dir=fig_dir,
                  title='{}_train'.format('1st'), vae=True)
    print(f'1st train MSE loss: {loss}')
    ## reconstruction validation
    # 1. our AE
    dataiter = iter(val_data_loader)
    val_img = dataiter.next()  # x.shape #torch.Size([128, 1, 28, 28])
    loss = resume('celeb256', val_img[0], z_dim=256, checkpoint_path=checkpoint_path_AE, fig_dir=fig_dir,
                  title='{}_val'.format('AE'), vae=False)
    print(f'AE val MSE loss: {loss}')
    # 2.two-stage VAE 1st
    loss = resume('celeb256', val_img[0], z_dim=256, checkpoint_path=checkpoint_path_1st, fig_dir=fig_dir,
                  title='{}_val'.format('1stAE'), vae=True)
    print(f'1st val MSE loss: {loss}')
