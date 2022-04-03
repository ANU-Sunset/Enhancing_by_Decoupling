import torch
import torch.nn as nn
from S2_Model import *
from utils import *

import os
from os.path import join
import torch.utils.data as data
import torch.optim as optim
from torch.utils.data import DataLoader


class zFromFile(data.Dataset):
    def __init__(self, z_list, root_path):
        super(zFromFile, self).__init__()
        self.root_path = root_path
        self.z_filenames = z_list

    def __getitem__(self, index):
        z = torch.load(join(self.root_path, self.z_filenames[index]))
        return z

    def __len__(self):
        return len(self.z_filenames)


def z_data(data_path='/scratch/u7076589/engn8536/Datasets/feature_map/DA_LOSS_training'):
    ## data
    z_list = [x for x in os.listdir(data_path)]

    # # TODO small test
    # num_train = 10
    # z_list = z_list[:num_train]
    print(f'train input z number:{len(z_list)}')  # 20000
    val_data_path = '/scratch/u7076589/engn8536/Datasets/feature_map/DA_LOSS_validation'
    val_z_list = [x for x in os.listdir(val_data_path)]
    print(f'validation input z number:{len(val_z_list)}')  # 2000

    train_set = zFromFile(z_list, data_path)
    val_set = zFromFile(val_z_list, val_data_path)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
    #                                    num_workers=num_workers)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
    #                                  num_workers=num_workers)
    # return train_loader, val_loader
    return train_set, val_set


def train_S2(train_data, val_data, start_epoch=0, num_epochs=1, dim=256, second_dim=1024, lr=2e-4, path='./'):
    # Model
    Model_S2 = S2Model(dim=dim, second_dim=second_dim).to(device)
    optimizer = optim.Adam(Model_S2.parameters(), lr=lr)
    print(Model_S2)
    results = {'train_rec_loss': [], 'train_KL_loss': [], 'train_loss': [], 'val_rec_loss': [], 'val_KL_loss': [],
               'val_loss': [], 'gamma': []}

    HALF_LOG_TWO_PI = 0.91893

    for epoch in range(start_epoch, num_epochs):
        Model_S2.train()

        ep_loss_rec = 0
        ep_loss_KL = 0

        for z in train_data:
            z = z.to(device)  # input
            mu, logvar, y, gamma = Model_S2(z)
            _, d = z.shape

            ## loss
            # reconstruction loss
            loss_rec = 0.5 * calc_reconstruction_loss(z, y) / gamma + d * torch.log(gamma) / 2 + d * HALF_LOG_TWO_PI
            # KL loss
            loss_KL = calc_kl(logvar, mu, reduce="mean")

            loss = loss_rec + loss_KL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for z in train_data:
            z = z.to(device)
            mu, logvar, y, gamma = Model_S2(z)
            _, d = z.shape

            ## loss
            # reconstruction loss
            loss_rec = 0.5 * calc_reconstruction_loss(z, y) / gamma + d * torch.log(gamma) / 2 + d * HALF_LOG_TWO_PI
            # KL loss
            loss_KL = calc_kl(logvar, mu, reduce="mean")

            ep_loss_rec += loss_rec.data.cpu().item() / len(train_data)
            ep_loss_KL += loss_KL.data.cpu().item() / len(train_data)
        results['train_rec_loss'].append(ep_loss_rec)
        results['train_KL_loss'].append(ep_loss_KL)
        results['train_loss'].append(ep_loss_rec + ep_loss_KL)

        #### Validation
        ep_val_loss_rec = 0
        ep_val_loss_kl = 0
        min_val_loss = 1e8
        Model_S2.eval()

        with torch.no_grad():
            for z in val_data:
                z = z.to(device)
                mu, logvar, y, gamma = Model_S2(z)
                ## loss
                # reconstruction loss
                _, d = z.shape
                loss_rec = 0.5 * calc_reconstruction_loss(z, y) / gamma + d * torch.log(gamma) / 2 + d * HALF_LOG_TWO_PI

                # KL loss
                loss_KL = calc_kl(logvar, mu, reduce="mean")

                ep_val_loss_kl += loss_KL.data.cpu().item() / len(val_data)
                ep_val_loss_rec += loss_rec.data.cpu().item() / len(val_data)
        #### save models
        if min_val_loss > (ep_val_loss_rec + ep_val_loss_kl):
            prefix = 'celeb256' + "_2nd_stage_" + "_VAE_gamma_"
            save_checkpoint(Model_S2, optimizer, gamma, prefix, path=path)
            min_val_loss = ep_val_loss_rec + ep_val_loss_kl
            print(f'save model at epoch#{epoch}')

        results['val_rec_loss'].append(ep_val_loss_rec)
        results['val_KL_loss'].append(ep_val_loss_kl)
        results['val_loss'].append(ep_val_loss_rec + ep_val_loss_kl)

        results['gamma'].append(gamma.data.cpu().item())
        plot_gamma(epoch, results['gamma'], path=path)  # plot gamma
        save_loss_plot(results, path=path, num_epochs=epoch)
        print(results)


if __name__ == '__main__':
    ## device
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dim = 256
    second_dim = 1024
    # batch_size=32
    # num_workers = 4
    path = './loss_plot'
    if not os.path.exists(path):
        os.makedirs(path)
    train_data, val_data = z_data()

    train_S2(train_data, val_data, start_epoch=0, num_epochs=1001, dim=dim, second_dim=second_dim, path=path)