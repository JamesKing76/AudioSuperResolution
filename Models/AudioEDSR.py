import math

import torch.nn as nn

from Models.Upsampler import Upsampler


# Biased on the tutorial which is based on https://arxiv.org/pdf/1707.02921.pdf

class ResidualBlock(nn.Module):
    def __init__(self, num_features=64, kernel_size=9, res_scale=0.1):
        """
        This creates a residual block
        :param num_features: The number of channels in a convolution
        :param kernel_size:  The size of each the kernel used in each convolution
        :param res_scale:    The amount the signal is scaled by
        """

        super(ResidualBlock, self).__init__()
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.block1 = nn.Sequential(
            nn.Conv1d(self.num_features, self.num_features, kernel_size, stride=1,
                      padding=math.ceil((self.kernel_size - 1) / 2)),
            # PhaseShuffle(2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.num_features, self.num_features, kernel_size, stride=1,
                      padding=math.ceil((self.kernel_size - 1) / 2)),
        )

    def forward(self, x):
        """
        Forward propagates through the block.
        :param x: Input features.
        :return:  Output of the ResBlock, of size (batch_size, num_features, input_length)
        """

        res = self.block1(x)
        res = res.mul(self.res_scale)
        res = res + x
        return res


class AudioEDSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_resblocks=32, res_scale=0.1,
                 upscale_factor=2, kernel_size=9):
        """
        Builds an AudioEDSR
        :param in_channels:    The number of input channels
        :param out_channels:   The number of output channels
        :param num_features:   The number of features the residual blocks learn over
        :param num_resblocks:  The number of residual blocks used in the network
        :param res_scale:      The amount the signal is scaled by in each residual block
        :param upscale_factor: The amount the signal is unscaled by
        :param kernel_size:    The kernel size of the convolutions
        """

        super(AudioEDSR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_resblocks = num_resblocks
        self.res_scale = res_scale
        self.kernel_size = kernel_size
        self.upscale_factor = upscale_factor
        self.build_network()

    def build_network(self):
        """
        Constructs the network.
        """

        block1 = [nn.Conv1d(self.in_channels, self.num_features, self.kernel_size, stride=1,
                            padding=math.ceil((self.kernel_size - 1) / 2))]

        block2 = [ResidualBlock(num_features=self.num_features, res_scale=self.res_scale) for _ in
                  range(self.num_resblocks)]
        block2.append(nn.Conv1d(self.num_features, self.num_features, self.kernel_size, stride=1,
                                padding=math.ceil((self.kernel_size - 1) / 2)))

        block3 = [Upsampler(self.upscale_factor, kernel_size=self.kernel_size, num_features=self.num_features),
                  nn.Conv1d(self.num_features, self.out_channels, self.kernel_size, stride=1,
                            padding=math.ceil((self.kernel_size - 1) / 2)), nn.ReLU()]

        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)

    def forward(self, x):
        """
        :param x: The input signal
        :return:  The output of the block
        """

        x = self.block1(x)
        res = self.block2(x)

        # adds the residual to the global output
        res = res + x
        fin = self.block3(res)

        # the bellow code was commented out to allow bitrate upsampling too
        # scaledup = torch.mul(fin, 2 ** 16)
        # fin[fin < 0] = 0
        # fin[fin > 2] = 0
        # las = torch.round(fin)
        # las = torch.mul(las, 2 ** -15)

        return fin
