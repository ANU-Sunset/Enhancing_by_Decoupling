import torch.nn as nn
import torch

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
    def __init__(self, cdim=3, zdim=256, channels=(64, 128, 256, 512, 512, 512), image_size=256):
        super(Encoder, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size

        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))

        self.fc = nn.Sequential(
                nn.Linear(cc * 4 * 4, zdim),
                # nn.Dropout(p=0.5),
                # nn.ReLU()
        )

    def forward(self, x):

        y = self.main(x).view(x.size(0), -1)
        # print(y.shape)
        y = self.fc(y)
        # print(y.shape)
        return y


class Decoder(nn.Module):
    """
    https://github.com/taldatech/soft-intro-vae-pytorch
    Difference: remove conditional and self.cond_dim, replace zdim into indim, forward function
    """

    def __init__(self, cdim=3, zdim=256, channels=(64, 128, 256, 512, 512, 512), image_size=256):
        super(Decoder, self).__init__()

        assert (2 ** len(channels)) * 4 == image_size
        cc = channels[-1]

        self.fc = nn.Sequential(
            nn.Linear(zdim, cc * 4 * 4),
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

    def forward(self, y):
        y = self.fc(y)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.main(y)
        return y


"""
Model
"""


class SoftIntroVAE(nn.Module):
    def __init__(self, cdim=3, zdim=256, channels=(64, 128, 256, 512, 512, 512), image_size=256):
        super(SoftIntroVAE, self).__init__()

        self.encoder = Encoder(cdim, zdim, channels, image_size)

        self.decoder = Decoder(cdim, zdim, channels, image_size)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y

    # def sample(self, z):
    #     return self.decode(z)
    #
    # def encode(self, x):
    #     return self.encoder(x)
    #
    # def decode(self, z):
    #     return self.decoder(z)