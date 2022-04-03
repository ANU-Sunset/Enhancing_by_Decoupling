import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
import os

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
    fig.savefig(path+"/training_rec_loss")  # TODO
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
        fig.savefig(path+"/training_KL_loss")  # TODO
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
        fig.savefig(path+"/training_loss")  # TODO
        plt.close()

def plot_gamma(num_epochs, results, title='gamma',path='./'):
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
    fig.savefig(path+"/gamma")  # TODO
    plt.close()