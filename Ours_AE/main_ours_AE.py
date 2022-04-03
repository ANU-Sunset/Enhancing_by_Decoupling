# Refer to Soft-Intro-Vae-2021 cvpr Encoder-Decoder
# https://github.com/taldatech/soft-intro-vae-pytorch

from train import *
from time import perf_counter

if __name__ == '__main__':
    """
    Recommended hyper-parameters:
    - CIFAR10: z_dim: 128, batch_size: 32
    - SVHN: z_dim: 128, batch_size: 32
    - MNIST: z_dim: 32, batch_size: 128
    - FashionMNIST: z_dim: 32, batch_size: 128
    - CelebA-HQ: z_dim: 256, batch_size: 8
    """
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    try:
        # checkpoint_path = '/scratch/u7076589/zdim256/celeb256_zdim_256_model.pth'

        batch_size = 8
        print(f'batch_size: {batch_size}')
        t_start = perf_counter()
        model, results = train_soft_intro_vae(dataset="celeb256", z_dim=256, batch_size=batch_size, num_workers=4,
                                              num_epochs=1001,
                                              device=device, start_epoch=0, lr=2e-4, test_iter=50, resume=False,
                                              checkpoint_path="")
        t_end = perf_counter()
        print(f'Training Time: {round((t_end - t_start) / 60, 2)} minutes')


    except SystemError:
        print("Error, probably loss is NaN, try again...")
