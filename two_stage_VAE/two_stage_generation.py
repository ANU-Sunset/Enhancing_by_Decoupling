from Model import *
from stage2.S2_Model import *
import torch
import torch.optim as optim
import torchvision.utils as vutils


def sample(model_s1, model_s2, num_samples, current_device, idx, latent_dim=256, fig_dir='./'):
    """
    Samples from the latent space and return the corresponding
    image space map.
    :param num_samples: (Int) Number of samples
    :param current_device: (Int) Device to run the model
    :return: (Tensor)
    """
    z = torch.randn(num_samples,latent_dim)
    z = z.to(current_device)

    z_prime = model_s2.decoder(z) # second stage decoder
    y = model_s1.decoder(z_prime) # first stage decoder
    max_imgs = min(num_samples, 4)
    vutils.save_image(
        torch.cat([y[:max_imgs]], dim=0).data.cpu(),
        '{}/gen_image_{}.jpg'.format(fig_dir,idx), nrow=4)
    # # return samples

if __name__ == '__main__':

    #---------------------------------------------------------------------------------
    ## device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #---------------------------------------------------------------------------------
    fig_dir='/scratch/u7076589/engn8536/Datasets/z_map/FID/two_stage_gen'  # FID folder
    #---------------------------------------------------------------------------------
    ## 1st stage
    # 1st model
    checkpoint_1st = '/scratch/u7076589/project_2stages/celeb256_zdim_256_VAE_gamma_model.pth'
    # build model
    channels = [64, 128, 256, 512, 512, 512]
    image_size = 256
    ch = 3

    model_s1 = VAEModel(cdim=ch, zdim=256, channels=channels, image_size=image_size, vae=True).to(device)
    optimizer_s1 = optim.Adam(model_s1.parameters(), lr=2e-4)
    # resume model
    print(f'Checkpoint{checkpoint_1st}')
    checkpoint = torch.load(checkpoint_1st)
    model_s1.load_state_dict(checkpoint['model'])
    optimizer_s1.load_state_dict(checkpoint['optimizer'])
    gamma1 = checkpoint['gamma']
    print(f'1st gamma: {gamma1.data.cpu().item()}')
    print('Resume 1st done.')
    model_s1.eval()
    #---------------------------------------------------------------------------------
    ## 2nd stage
    # 2nd model
    checkpoint_2nd = '/scratch/u7076589/project_2stages/stage2/2nd_stage_loss_plot/celeb256_2nd_stage__VAE_gamma_model.pth'
    # build model
    model_s2 = S2Model(dim=256,second_dim=1024).to(device)
    optimizer_s2 = optim.Adam(model_s2.parameters(), lr=2e-4)
    # resume model
    print(f'Checkpoint{checkpoint_2nd}')
    checkpoint = torch.load(checkpoint_2nd)
    model_s2.load_state_dict(checkpoint['model'])
    optimizer_s2.load_state_dict(checkpoint['optimizer'])
    gamma2 = checkpoint['gamma']
    print(f'2nd gamma: {gamma2.data.cpu().item()}')
    print('Resume 2st done.')
    model_s2.eval()
    #---------------------------------------------------------------------------------
    # generation
    # 1024-dim normal distribution -> S2 decoder -> S1 decoder
    num_samples=200
    for i in range(num_samples): # save one by one for FID calculation
        sample(model_s1, model_s2, num_samples=1, current_device=device, idx=i, latent_dim=256, fig_dir=fig_dir)