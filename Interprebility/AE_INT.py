import torch
import os
from Net import Net
import torchvision.utils as vutils
from Encoder_Decoder_Model import *
import torch.optim as optim
from copy import deepcopy
from get_diff_dim import get_diff_dim
from dataset import NetData

torch.manual_seed(34)  # rotation good ppt1
# torch.manual_seed(46) # left face good quality ppt2
# torch.manual_seed(54)  # good quality front face ppt3

# torch.manual_seed(51) # right face good quality  ppt del
# torch.manual_seed(9)
# torch.manual_seed(6)
# torch.manual_seed(22) # rotation
# torch.manual_seed(26) # good quality
# torch.manual_seed(28) # rotation
# torch.manual_seed(44) # front face good quality
# torch.manual_seed(46) # left face good quality
# torch.manual_seed(50) # left face good quality
# torch.manual_seed(57)  # good quality front face
# torch.manual_seed(58) # eye rolling
# torch.manual_seed(60) # front good quality
# torch.manual_seed(70)


import torch
import os


def get_diff_dim(input_size, changed_dim, whole_files, net, mu, var):
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    file_list = '../../feature_map/DA_LOSS_training'
    # input_size=1,changed_dim=74 means output 256 first image with diff dim
    items = None
    count = 0
    for pt_file in whole_files:
        if items == None:
            items = torch.load(file_list + '/' + pt_file)
        else:
            items = torch.cat((items, torch.load(file_list + '/' + pt_file)), 0)
        count += 1
        if count == input_size:
            break
    print(pt_file)
    items = items.to(device)
    items = (items - mu) / var

    z = net(items).to(device)  # 20*256
    for i in range(z.size(0)):
        one = torch.zeros([6, 256]).to(device)
        one[0] = z[i]
        one[1] = z[i]
        one[2] = z[i]
        one[3] = z[i]
        one[4] = z[i]
        one[5] = z[i]
        one[1, changed_dim] = one[1, changed_dim] * -1
        one[2, changed_dim] = one[2, changed_dim] * -2
        one[3, changed_dim] = one[3, changed_dim] * -3
        one[4, changed_dim] = one[4, changed_dim] * -4
        one[5, changed_dim] = one[5, changed_dim] * -6
        z1 = net.inverse(one)
        z1 = z1 * var + mu
        # path = '../yona_final/'+ str(changed_dim)+'.pt' # for single img
        path = '../yona_final/' + str(i) + '_' + str(changed_dim) + '.pt'  # for 20 img
        torch.save(z1, path)


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------
    ## device
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    # ---------------------------------------------------------------------------------
    # iResNet
    file_list = '../../feature_map/DA_LOSS_training'
    files = os.listdir(file_list)
    items = None
    for pt_file in files:
        if items == None:
            items = torch.load(file_list + '/' + pt_file)
        else:
            items = torch.cat((torch.load(file_list + '/' + pt_file), items), 0)
    mu = (sum(items) / items.size(0)).to(device)
    var = torch.sqrt(torch.var(items, dim=0)).to(device)
    # print(mu)
    # print(var)

    resume_checkpoint_path = 'model.pt'
    net = torch.load(resume_checkpoint_path).to(device)

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
    checkpoint_path = '/scratch/u7076589/project_DA_loss/celeb256_zdim_256_DA_lossmodel.pth'
    print(f'Checkpoint{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Resume done.')
    model.eval()

    # vae_path = '/scratch/u7076589/project_VAE/celebA/celeb256_zdim_256_VAE_gamma_0.5_model.pth'
    # checkpoint = torch.load(vae_path)

    # ---------------------------------------------------------------------------------
    ## reconstruction
    net_data = NetData()
    mu, var = net_data.get_mu_var()
    mu = mu.to(device)
    var = var.to(device)
    resume_checkpoint_path = 'model.pt'
    net = torch.load(resume_checkpoint_path).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    file_list = '../../feature_map/DA_LOSS_training'
    files = os.listdir(file_list)
    line = None
    print("begin loop")
    # find 256 dims of first image
    """
    for i in range(256):
        get_diff_dim(1,i,files,net,mu,var)
        z=torch.load('/scratch/u7076589/engn8536/Datasets/z_map/yona_final/{}.pt'.format(i))
        # print(z[0].shape)
        z=z.to(device)
        rec_img = model.decoder(z[0].view(1,256))
        rec_img_0 = model.decoder(z[1].view(1,256))
        rec_img_1 = model.decoder(z[2].view(1,256))
        rec_img_2 = model.decoder(z[3].view(1,256))
        rec_img_3 = model.decoder(z[4].view(1,256))
        rec_img_4 = model.decoder(z[5].view(1,256))
        one = torch.cat([rec_img,rec_img_0,rec_img_1,rec_img_2,rec_img_3,rec_img_4],dim=0)
        if line == None:
            line = one
        else:
            line = torch.cat((line,one),0)
        print(line.size())
        if (i+1)%8==0:
            vutils.save_image(
                line, 
                '{}/rec_image_{}.jpg'.format('yona_figure_rec',i), nrow=6)
            line = None
    """
    # find 73, 116, 182, 219, 21, 190, 232, 150, 126, 40 dim of first 20 :
    get_diff_dim(20, 40, files, net, mu, var)
    for i in range(20):
        z = torch.load('/scratch/u7076589/engn8536/Datasets/z_map/yona_final/{}_40.pt'.format(i))  # remember to modify
        z = z.to(device)
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
        if (i + 1) % 5 == 0:
            vutils.save_image(
                line,
                '{}/rec_image_40_{}.jpg'.format('yona_figure_rec', i), nrow=6)
            line = None

