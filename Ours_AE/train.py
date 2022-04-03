from Encoder_Decoder_Model import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from helper import *
from time import perf_counter
import os
import random
from dataset import *
from tqdm import tqdm
import torchvision.utils as vutils

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

def save_loss_plot(train_loss, val_loss, start_epoch=0, num_epochs=10):
    fig, ax = plt.subplots(figsize=[8, 6])
    tl_line1 = ax.plot(train_loss, label='training loss', color='red')
    vl_line2 = ax.plot(val_loss, label='validation loss', color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss with epoch#{num_epochs}')
    lines = tl_line1 + vl_line2
    labels = [_.get_label() for _ in lines]
    ax.legend(lines, labels, loc='center right')
    if num_epochs<10:
        plt.xticks([i for i in range(num_epochs)])
    else:
        plt.xticks([i*(num_epochs//10) for i in range(num_epochs//(num_epochs//10)+1)])
    plt.xlim(xmin=start_epoch)
    fig.savefig("/scratch/u7076589/project_DA_loss/training_loss")


def save_checkpoint(model, optimizer, epoch, prefix=""):
    model_out_path = "/scratch/u7076589/project_DA_loss/" + prefix + "model.pth"
    state = {"epoch": epoch, "model": model.state_dict(),"optimizer":optimizer.state_dict()}
    if not os.path.exists("/scratch/u7076589/project_DA_loss"):
        os.makedirs("/scratch/u7076589/project_DA_loss")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def train_soft_intro_vae(dataset='cifar10', z_dim=512, batch_size=128, num_workers=4, lr= 2e-4,
                             start_epoch=0,
                             num_epochs=10, recon_loss_type="mse", num_row=8,
                             seed=-1, device=torch.device("cpu"),test_iter=1000,resume=False,checkpoint_path=''):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)
    # --------------build dataset -------------------------
    if  dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
        output_height = 256
        # TODO small test
        # train_size = 162770
        train_size = 20000
        #train_size = 10
        # data_root = 'data256x256'
        data_root = '/scratch/engn8536/Datasets/data256x256'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        print(f'images: {len(image_list)}')
        train_list = image_list[:train_size]
        print(f"train_images size: {train_size}")
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
        ## validation
        val_size = 2000
        print(f"validation_images size: {val_size}")
        val_list = image_list[train_size:train_size+val_size]
        val_set = ImageDatasetFromFile(val_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True,train=False)

    else:
        raise NotImplementedError("dataset is not supported")

    # --------------build models -------------------------
    model = SoftIntroVAE(cdim=ch, channels=channels, image_size=image_size).to(device)

    fig_dir = '/scratch/u7076589/project_DA_loss/figures_' + dataset + '_zdim_'+str(z_dim)
    os.makedirs(fig_dir, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)

    # print('model')
    # print(model.state_dict())
    # print('optimizer')
    # print(optimizer.state_dict())

    if resume:
        print(f'Checkpoint{checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)

        # print(checkpoint['model'])
        # print(checkpoint['optimizer'])

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resume training.')
    scale = 1 / (ch * image_size ** 2)  # normalize by images size (channels * height * width)

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    # start_time = perf_counter()

    results = {'train_loss':[],'val_loss':[]}
    cur_iter = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()

        # pbar = tqdm(iterable=train_data_loader)
        ep_loss = 0

        # for batch in pbar:
        for batch, batch_erase in train_data_loader:
            # --------------train------------
            # soft-intro-vae training loss reduces into AE_testing_experiment loss only

            # real data
            real_batch = batch.to(device)
            # erased data
            erased_batch = batch_erase.to(device)
            z_real = model.encoder(erased_batch) # encoder
            rec = model.decoder(z_real) # decoder

            # AE_testing_experiment loss
            # real
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            # backprop
            optimizer.zero_grad()
            loss_rec.backward()
            optimizer.step()

            if torch.isnan(loss_rec):
                raise SystemError
            # pbar.set_description_str('epoch #{}'.format(epoch))
            # pbar.set_postfix(loss=loss_rec.data.cpu().item())

        for batch, batch_erase in train_data_loader:
            # real data
            real_batch = batch.to(device)
            # erased data
            erased_batch = batch_erase.to(device)
            z_real = model.encoder(erased_batch) # encoder
            rec = model.decoder(z_real) # decoder

            # AE_testing_experiment loss
            # real
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

            ep_loss+= loss_rec.data.cpu().item()/len(train_data_loader)

        results['train_loss'].append(ep_loss)
        # pbar.close()
        # save reconstructin images
        if cur_iter % test_iter == 0:
            _, rec_det = model(erased_batch) # return z, reconstructed y
            max_imgs = min(batch.size(0), 16)
            vutils.save_image(
            torch.cat([erased_batch[:max_imgs], rec_det[:max_imgs]], dim=0).data.cpu(), '{}/train_rec_image_{}.jpg'.format(fig_dir, cur_iter+start_epoch), nrow=num_row)


        #### validation
        ep_val_loss = 0
        min_val_loss = 1e8
        with torch.no_grad():
            for batch, batch_erase in val_data_loader:
                # real data
                real_batch = batch.to(device)
                z_real = model.encoder(real_batch) # encoder
                rec = model.decoder(z_real) # decoder

                # AE_testing_experiment loss
                # real
                loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
                ep_val_loss+= loss_rec.data.cpu().item()/len(val_data_loader)
        results['val_loss'].append(ep_val_loss)
        # save validation AE_testing_experiment images
        if cur_iter % test_iter == 0:
            _, rec_det = model(real_batch) # return z, reconstructed y
            max_imgs = min(batch.size(0), 16)
            vutils.save_image(
            torch.cat([real_batch[:max_imgs], rec_det[:max_imgs]], dim=0).data.cpu(), '{}/val_rec_image_{}.jpg'.format(fig_dir, cur_iter+start_epoch), nrow=num_row)

        cur_iter += 1

        # save models
        if min_val_loss> ep_val_loss:
            prefix = dataset + "_zdim_" + str(z_dim) +"_" + "DA_loss"
            save_checkpoint(model, optimizer, epoch, prefix)
            min_val_loss = ep_val_loss
            print(f'save model at epoch#{epoch}')

        save_loss_plot(results['train_loss'],results['val_loss'], start_epoch=start_epoch, num_epochs=epoch+start_epoch)
        print(results)

    # print(f'training time: {perf_counter()-start_time:.2f}s', )
    return model, results
