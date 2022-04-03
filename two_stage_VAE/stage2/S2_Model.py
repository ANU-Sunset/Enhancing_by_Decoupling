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


class s2_Encoder(nn.Module):
    def __init__(self, dim=256, second_dim=1024):
        super(s2_Encoder, self).__init__()
        self.dim = dim
        # self.second_depth=second_depth
        self.second_dim = second_dim

        self.main = nn.Sequential(  # second_depth=3
            nn.Linear(self.dim, self.second_dim),
            nn.ReLU(),
            nn.Linear(self.second_dim, self.second_dim),
            nn.ReLU(),
            nn.Linear(self.second_dim, self.second_dim),
            nn.ReLU()
        )

        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]

        self.fc = nn.Linear(num_fc_features, 2 * dim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.dim)
        dummy_input = torch.cat((dummy_input, self.main(dummy_input)), -1)
        return dummy_input[0].shape

    def forward(self, z):
        y = self.main(z)
        y = torch.cat((z, y), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class s2_Decoder(nn.Module):
    def __init__(self, dim=256, second_dim=1024):
        super(s2_Decoder, self).__init__()
        self.dim = dim
        self.second_dim = second_dim
        self.main = nn.Sequential(  # second_depth=3
            nn.Linear(self.dim, self.second_dim),
            nn.ReLU(),
            nn.Linear(self.second_dim, self.second_dim),
            nn.ReLU(),
            nn.Linear(self.second_dim, self.second_dim),
            nn.ReLU()
        )
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]

        self.fc = nn.Linear(num_fc_features, self.dim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.dim)
        dummy_input = torch.cat((dummy_input, self.main(dummy_input)), -1)
        return dummy_input[0].shape

    def forward(self, z):
        y = self.main(z)
        y = torch.cat((z, y), -1)
        y = self.fc(y)
        return y


class S2Model(nn.Module):
    def __init__(self, dim=256, second_dim=1024):
        super(S2Model, self).__init__()
        self.dim = dim
        self.second_dim = second_dim
        self.encoder = s2_Encoder(dim=self.dim, second_dim=self.second_dim)
        self.decoder = s2_Decoder(dim=self.dim, second_dim=self.second_dim)
        self.loggamma2 = torch.nn.parameter.Parameter(torch.ones(1))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        y = self.decoder(z)
        return mu, logvar, y, torch.exp(self.loggamma2)
