import torch
import torchvision.utils as vutils
from Encoder_Decoder_Model import *
import torch.optim as optim


if __name__ == '__main__':
    #---------------------------------------------------------------------------------
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #---------------------------------------------------------------------------------
    ## build model
    channels = [64, 128, 256, 512, 512, 512]
    image_size = 256
    ch = 3
    model = SoftIntroVAE(cdim=ch, zdim=256, channels=channels, image_size=image_size).to(device)
    # print(model.encoder)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    #---------------------------------------------------------------------------------
    ## resume model
    checkpoint_path = '/scratch/u7076589/project_DA_loss/celeb256_zdim_256_DA_lossmodel.pth'
    print(f'Checkpoint{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Resume done.')
    model.eval()

    input = torch.load('/scratch/u7076589/engn8536/Datasets/z_map/newc_27.pt') # TODO: pt file
    print(input.shape)

    #---------------------------------------------------------------------------------
    ## AE_testing_experiment
    # print(model.decoder)
    rec_img = model.decoder(input[:10])
    max_imgs = min(input.size(0), 16)

    vutils.save_image(
        torch.cat([rec_img[:max_imgs]],dim=0), '{}/iResnet_rec_image_12.jpg'.format('/scratch/u7076589/project_DA_loss/iResNet_Recon'), nrow=2)
