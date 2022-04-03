from time import perf_counter
from Encoder_Decoder_Model import *
import torchvision.transforms as transforms
import torch.optim as optim
import os
from os.path import join
from dataset import load_image
import torch
import torchvision.utils as vutils


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


if __name__ == '__main__':
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # ---------------------------------------------------------------------------------
    ## build model
    channels = [64, 128, 256, 512, 512, 512]
    image_size = 256
    ch = 3
    model = SoftIntroVAE(cdim=ch, zdim=256, channels=channels, image_size=image_size).to(device)
    # print(model.encoder)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # ---------------------------------------------------------------------------------
    ## resume model
    # checkpoint_path = '/scratch/u7076589/project/training/celeb256_zdim_512_resume_from_425_model.pth'
    # checkpoint_path = '/scratch/u7076589/project_DA/celeb256_zdim_256_DA_lossmodel.pth'
    # checkpoint_path = '/scratch/u7076589/project_DA/save.pth'
    checkpoint_path = '/scratch/u7076589/project_DA_loss/celeb256_zdim_256_DA_lossmodel.pth'
    print(f'Checkpoint{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Resume done.')
    model.eval()

    # model.encoder
    print(model.encoder)
    # model.decoder
    print(model.decoder)
    # ---------------------------------------------------------------------------------
    ## transform
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trans_erase_transform = transforms.RandomErasing(p=0.5, scale=(0.02, 0.05), ratio=(1.0, 1.1), value='random',
                                                     inplace=False)
    # ---------------------------------------------------------------------------------
    ## data input
    data_root = '/scratch/engn8536/Datasets/data256x256'
    image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
    print(f'images: {len(image_list)}')
    image_name = image_list[0]
    print(image_name)
    print(image_name.split('.')[0])
    # img_real = load_image(join(data_root, image_name),input_height=None,
    # output_height=256)
    # ---------------------------------------------------------------------------------
    # training set
    train_size = 20000
    train_list = image_list[:train_size]
    print(f"train_images size: {train_size}")

    for img_name in train_list:
        img = load_image(join(data_root, img_name), input_height=None, output_height=256)
        img = train_transform(img)
        # trans_erase_transform(img)  # erase effect
        c, h, w = img.shape
        img = img.view(1, c, h, w).to(device)

        ## feature map
        z = model.encoder(img)
        torch.save(z, '/scratch/u7076589/project_DA_loss/DA_LOSS_training/{}.pt'.format(img_name.split('.')[0]))
    # ---------------------------------------------------------------------------------
    ## validation set
    val_size = 2000
    print(f"validation_images size: {val_size}")
    val_list = image_list[train_size:train_size + val_size]

    for img_name in val_list:
        img = load_image(join(data_root, img_name), input_height=None, output_height=256)
        img = train_transform(img)
        c, h, w = img.shape
        img = img.view(1, c, h, w).to(device)

        ## feature map
        z = model.encoder(img)
        torch.save(z, '/scratch/u7076589/project_DA_loss/DA_LOSS_validation/{}.pt'.format(img_name.split('.')[0]))

    ### model input output image
    # img_real = input_transform(img_real)
    # img = trans_erase_transform(img_real)  # erase effect

    # c,h,w = img.shape
    # img = img.view(1,c,h,w).to(device)

    # print(img.shape)  # (1, 3, 256, 256)
    # output = model.encoder(img)
    # print(output)
    ### save feature map
    # torch.save(output,'/scratch/u7076589/project_500/encoder.pt')
    # o = torch.load('./zdim256epoch1000/encoder.pt')
    # print(o)

    ### print model
    # print(model.encoder)
    # print(model.decoder)

    # AE_testing_experiment
    # z, rec_img = model(img)
    # vutils.save_image(
    #     torch.cat([img.view(1,c,h,w).to(device),rec_img],dim=0), '{}/train_rec_image_{}.jpg'.format('/scratch/u7076589/project_DA/debug','010'), nrow=2)

    # print(output.shape)




