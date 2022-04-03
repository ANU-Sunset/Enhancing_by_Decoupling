from Model import *  #VAE model but with trainable gamma
import torch.utils.data as data
from os.path import join
from PIL import Image

import torch
import torchvision.transforms as transforms
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


def save_loss_plot(results, path='./', num_epochs=10, start_epoch=0, vae=True):
    # reconstruction loss
    fig, ax = plt.subplots(figsize=[8, 6])
    tl_line1 = ax.plot(results['train_rec_loss'], label='training reconstruction loss', color='red')
    vl_line1 = ax.plot(results['val_rec_loss'], label='validation reconstruction loss', color='red', linestyle='dashed')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title(f'Training Reconstruction Loss with epoch#{num_epochs}')
    lines = tl_line1 + vl_line1
    labels = [_.get_label() for _ in lines]
    ax.legend(lines, labels, loc='center right')
    if num_epochs < 10:
        plt.xticks([i for i in range(num_epochs)])
    else:
        plt.xticks([i * (num_epochs // 10) for i in range(num_epochs // (num_epochs // 10) + 1)])
    plt.xlim(xmin=start_epoch)
    fig.savefig(path + "/training_rec_loss")  # TODO
    plt.close()

    if vae:
        ### KL loss
        fig, ax = plt.subplots(figsize=[8, 6])
        tl_line2 = ax.plot(results['train_KL_loss'], label='training KL loss', color='blue')
        vl_line2 = ax.plot(results['val_KL_loss'], label='validation KL loss', color='blue', linestyle='dashed')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Loss')
        ax.set_title(f'Training KL Loss with epoch#{num_epochs}')
        lines = tl_line2 + vl_line2
        labels = [_.get_label() for _ in lines]
        ax.legend(lines, labels, loc='center right')
        if num_epochs < 10:
            plt.xticks([i for i in range(num_epochs)])
        else:
            plt.xticks([i * (num_epochs // 10) for i in range(num_epochs // (num_epochs // 10) + 1)])
        plt.xlim(xmin=start_epoch)
        fig.savefig(path + "/training_KL_loss")  # TODO
        plt.close()

        ### loss
        fig, ax = plt.subplots(figsize=[8, 6])

        tl_line3 = ax.plot(results['train_loss'], label='training loss', color='black')
        vl_line3 = ax.plot(results['val_loss'], label='validation loss', color='black', linestyle='dashed')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Loss with epoch#{num_epochs}')
        lines = tl_line3 + vl_line3
        labels = [_.get_label() for _ in lines]
        ax.legend(lines, labels, loc='center right')
        if num_epochs < 10:
            plt.xticks([i for i in range(num_epochs)])
        else:
            plt.xticks([i * (num_epochs // 10) for i in range(num_epochs // (num_epochs // 10) + 1)])
        plt.xlim(xmin=start_epoch)
        fig.savefig(path + "/training_loss")  # TODO
        plt.close()


def plot_gamma(num_epochs, results, title='gamma', path='./'):
    print(results[num_epochs])
    fig, ax = plt.subplots(figsize=[8, 6])
    tl_line1 = ax.plot(results, color='red')

    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.set_title(f'Gamma at {num_epochs}')
    lines = tl_line1

    if num_epochs < 10:
        plt.xticks([i for i in range(num_epochs)])
    else:
        plt.xticks([i * (num_epochs // 10) for i in range(num_epochs // (num_epochs // 10) + 1)])
    plt.xlim(xmin=0)
    fig.savefig(path + "/gamma")  # TODO
    plt.close()


def save_checkpoint(model, optimizer, gamma, prefix="", path='./'):  # TODO
    model_out_path = path + "/" + prefix + "model.pth"
    state = {"gamma": gamma, "model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict())}
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


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


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def train_vae(dataset='celeb256', path='./', z_dim=128, batch_size=32, num_workers=4, lr=2e-4,
              start_epoch=0,
              num_epochs=10, num_row=8,
              seed=-1, device=torch.device("cpu"), test_iter=1000, vae=True):
    if dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
        output_height = 256
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

    else:
        raise NotImplementedError("dataset is not supported")

    ### save location
    fig_dir = path + '/figures_' + dataset  # TODO
    os.makedirs(fig_dir, exist_ok=True)

    #### Model Optimizer
    model = VAEModel(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size, vae=vae).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(model)
    results = {'train_rec_loss': [], 'train_KL_loss': [], 'train_loss': [], 'val_rec_loss': [], 'val_KL_loss': [],
               'val_loss': [], 'gamma': []}

    cur_iter = 0
    HALF_LOG_TWO_PI = 0.91893
    #### Train
    for epoch in range(start_epoch, num_epochs):
        model.train()

        ep_loss_rec = 0
        ep_loss_KL = 0

        # train
        for x in train_loader:
            x = x.to(device)
            # print(x.shape) #torch.Size([128, 1, 28, 28])
            # forward
            mu, logvar, z, y, gamma = model(x)
            ## loss
            # reconstruction loss
            # recon_loss_type="mse"
            n, c, h, w = x.shape
            loss_rec = 0.5 * calc_reconstruction_loss(x, y) / gamma + h * w * torch.log(
                gamma) / 2 + h * w * HALF_LOG_TWO_PI
            if vae:
                # KL loss
                loss_KL = calc_kl(logvar, mu, reduce="mean")
                loss = loss_rec + loss_KL
            else:
                loss = loss_rec
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # calculate and save loss
        for x in train_loader:
            x = x.to(device)
            mu, logvar, z, y, gamma = model(x)
            ## loss
            # reconstruction loss
            n, c, h, w = x.shape
            loss_rec = 0.5 * calc_reconstruction_loss(x, y) / gamma + h * w * torch.log(
                gamma) / 2 + h * w * HALF_LOG_TWO_PI

            if vae:
                # kl loss
                loss_KL = calc_kl(logvar, mu, reduce="mean")
                ep_loss_KL += loss_KL.data.cpu().item() / len(train_loader)
            ep_loss_rec += loss_rec.data.cpu().item() / len(train_loader)

        results['gamma'].append(gamma.data.cpu().item())  # log gamma value
        plot_gamma(epoch, results['gamma'])  # plot gamma
        results['train_rec_loss'].append(ep_loss_rec)
        if vae:
            results['train_KL_loss'].append(ep_loss_KL)
            results['train_loss'].append(ep_loss_rec + ep_loss_KL)
        # save reconstruction images
        if cur_iter % test_iter == 0:
            _, _, _, rec_det, _ = model(x)
            if dataset == 'celeb256':
                max_imgs = min(x.size(0), 16)
            else:
                max_imgs = min(x.size(0), 4)
            vutils.save_image(
                torch.cat([x[:max_imgs], rec_det[:max_imgs]], dim=0).data.cpu(),
                '{}/train_rec_image_{}.jpg'.format(fig_dir, cur_iter + start_epoch), nrow=num_row)
        #### Validation
        ep_val_loss_rec = 0
        ep_val_loss_kl = 0
        min_val_loss = 1e8
        model.eval()
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                mu, logvar, z, y, gamma = model(x)
                ## loss
                # reconstruction loss
                n, c, h, w = x.shape
                # gamma = torch.exp(loggamma)
                loss_rec = 0.5 * calc_reconstruction_loss(x, y) / gamma + h * w * torch.log(
                    gamma) / 2 + h * w * HALF_LOG_TWO_PI
                if vae:
                    # KL loss
                    loss_KL = calc_kl(logvar, mu, reduce="mean")
                    loss = loss_rec + loss_KL
                    ep_val_loss_kl += loss_KL.data.cpu().item() / len(val_loader)
                ep_val_loss_rec += loss_rec.data.cpu().item() / len(val_loader)

        results['val_rec_loss'].append(ep_val_loss_rec)
        if vae:
            results['val_KL_loss'].append(ep_val_loss_kl)
            results['val_loss'].append(ep_val_loss_rec + ep_val_loss_kl)
        ##### save validation reconstruction images
        if cur_iter % test_iter == 0:
            _, _, _, rec_det, _ = model(x)
            if dataset == 'celeb256':
                max_imgs = min(x.size(0), 16)
            else:
                max_imgs = min(x.size(0), 4)
            vutils.save_image(
                torch.cat([x[:max_imgs], rec_det[:max_imgs]], dim=0).data.cpu(),
                '{}/val_rec_image_{}.jpg'.format(fig_dir, cur_iter + start_epoch), nrow=num_row)

        cur_iter += 1

        #### save models
        if min_val_loss > (ep_val_loss_rec + ep_val_loss_kl):
            prefix = dataset + "_zdim_" + str(z_dim) + "_VAE_gamma_"
            save_checkpoint(model, optimizer, gamma, prefix, path=path)
            min_val_loss = ep_val_loss_rec + ep_val_loss_kl
            print(f'save model at epoch#{epoch}')

        save_loss_plot(results, path=path, num_epochs=epoch, vae=vae)
        print(results)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    try:
        t_start = perf_counter()

        # VAE celebA
        train_vae(dataset='celeb256', z_dim=256, batch_size=8, num_workers=4, lr=2e-4,
                  start_epoch=0, num_epochs=1001, device=device,
                  num_row=8, test_iter=50)

        t_end = perf_counter()
        print(f'Training Time: {round((t_end - t_start) / 60, 2)} minutes')

    except SystemError:
        print("Error, probably loss is NaN, try again...")
