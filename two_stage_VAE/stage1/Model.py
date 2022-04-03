import torch.nn as nn
import torch

"""
Helper function
"""


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


"""
Model Blocks
"""


class _Residual_Block(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        # output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


class Encoder(nn.Module):
    """
    https://github.com/taldatech/soft-intro-vae-pytorch
    Difference: y on output, not (mu, logvar), remove conditional, replace zdim into outdim, forward function
    """

    def __init__(self, cdim=1, zdim=32, channels=(64, 128), image_size=28, vae=True):
        super(Encoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )
        self.vae = vae

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))

        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        # print("conv shape: ", self.conv_output_size)
        # print("num fc features: ", num_fc_features)
        if self.vae:
            self.fc = nn.Linear(num_fc_features, 2 * zdim)
        else:  # autoencoder
            self.fc = nn.Linear(num_fc_features, zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        # print(y.shape)
        y = self.fc(y)
        if self.vae:
            mu, logvar = y.chunk(2, dim=1)
        else:
            return y, y
        return mu, logvar


class Decoder(nn.Module):
    """
    https://github.com/taldatech/soft-intro-vae-pytorch
    Difference: remove conditional and self.cond_dim, replace zdim into indim, forward function
    """

    def __init__(self, cdim=1, zdim=32, channels=(64, 128), image_size=28, conv_input_size=None):
        super(Decoder, self).__init__()
        self.conv_input_size = conv_input_size
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]

        self.fc = nn.Sequential(
            nn.Linear(zdim, num_fc_features),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        # z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


"""
Model
"""


class VAEModel(nn.Module):
    def __init__(self, cdim=1, zdim=32, channels=(64, 128), image_size=28, vae=True):
        super(VAEModel, self).__init__()
        self.vae = vae
        self.encoder = Encoder(cdim, zdim, channels, image_size, self.vae)

        self.decoder = Decoder(cdim, zdim, channels, image_size, conv_input_size=self.encoder.conv_output_size)
        self.loggamma = torch.nn.parameter.Parameter(torch.ones(1))  # TODO gamma

    def forward(self, x):
        mu, logvar = self.encoder(x)
        if self.vae:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        y = self.decoder(z)

        return mu, logvar, z, y, torch.exp(self.loggamma)