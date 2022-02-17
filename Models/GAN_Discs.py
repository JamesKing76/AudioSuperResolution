import torch
import torch.nn as nn


# Builds a Discriminator (biased on https://arxiv.org/pdf/1609.04802.pdf)

class Discriminator(nn.Module):
    def __init__(self, DiscBlocks, in_channels=1, inital_filter_size=8):
        """
        Builds a discriminator block.
        :param DiscBlocks:  The number of blocks in the discriminator
        :param in_channels: The number of input channels
        """

        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.DiscBlocks = DiscBlocks
        self.inital_filter_size = inital_filter_size
        self.build_discriminator()

    def build_discriminator(self):
        """
        Constructs the discriminator network
        """

        n = self.DiscBlocks
        m = self.inital_filter_size
        block1 = [nn.Conv1d(self.in_channels, m, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.2)]

        block2 = [nn.Conv1d(m, m, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2)]

        block3 = []

        for i in range(n):
            prev = m * (1 << i)
            next = m * (1 << (i + 1))
            block3.append(nn.Conv1d(prev, next, kernel_size=3, padding=1))
            block3.append(nn.LeakyReLU(0.2))
            # block3.append(PhaseShuffle(2))

            block3.append(nn.Conv1d(next, next, kernel_size=3, stride=2, padding=1))
            block3.append(nn.LeakyReLU(0.2))
            # block3.append(PhaseShuffle(2))

        block4 = [nn.AdaptiveAvgPool1d(1),
                  nn.Conv1d(m * (2 ** n), m * (2 ** (n + 1)), kernel_size=1),
                  nn.LeakyReLU(0.2),
                  nn.Conv1d(m * (2 ** (n + 1)), 1, kernel_size=1)]

        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)

    def forward(self, x):
        """
        Forward propagates through the network.
        :param x: input image.
        :return: binary classification of image, either fake (0) or real (1).
        """
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.nn.functional.leaky_relu(x.reshape(batch_size), negative_slope=1)
