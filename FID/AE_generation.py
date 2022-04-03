import torch
from Net import Net
import matplotlib.pyplot as plt
import torch
import os
from Net import Net
import torchvision.utils as vutils
from Encoder_Decoder_Model import *
import torch.optim as optim
from copy import deepcopy

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # ---------------------------------------------------------------------------------
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    resume_checkpoint_path = 'model.pt'
    net = torch.load(resume_checkpoint_path).cuda()
    # net_data = NetData().cuda()

    # ---------------------------------------------------------------------------------
    ## build model
    channels = [64, 128, 256, 512, 512, 512]
    image_size = 256
    ch = 3
    model = SoftIntroVAE(cdim=ch, zdim=256, channels=channels, image_size=image_size).to(device)
    # print(model.encoder)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # ---------------------------------------------------------------------------------
    ## read datas
    file_list = '../../feature_map/DA_LOSS_training'
    files = os.listdir(file_list)
    items = None
    for pt_file in files:
        if items == None:
            items = torch.load(file_list + '/' + pt_file)
        else:
            items = torch.cat((torch.load(file_list + '/' + pt_file), items), 0)
    mu = sum(items) / items.size(0)
    var = torch.sqrt(torch.var(items, dim=0))

    ## resume model
    checkpoint_path = '/scratch/u7076589/project_DA_loss/celeb256_zdim_256_DA_lossmodel.pth'
    print(f'Checkpoint{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Resume done.')
    model.eval()

    # for i in range(200):
    #     print(i)
    #     inverse_input = torch.randn(1, 256).cuda()
    #     inverse_output = net.inverse(inverse_input)
    #     inverse_output = inverse_output * var + mu
    #     z = inverse_output
    #     rec_img = model.decoder(z)
    #     vutils.save_image(
    #         rec_img,
    #         '{}/iResnet_rec_image_{}.jpg'.format('/scratch/u7076589/engn8536/Datasets/z_map/FID/our_gen',i), nrow=7)
    inverse_input = torch.randn(200, 256).cuda()
    inverse_output = net.inverse(inverse_input)
    inverse_output = inverse_output * var + mu
    z = inverse_output
    rec_img = model.decoder(z)
    for i in range(200):
        vutils.save_image(
            rec_img[i],
            '{}/iResnet_rec_image_{}.jpg'.format('/scratch/u7076589/engn8536/Datasets/z_map/FID/our_gen', i), nrow=1)
