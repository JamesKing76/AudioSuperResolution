import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.Upsampler import PixelShuffle1d, PhaseShuffle


# Models in this file are based of the AudioUNet structure, as introduced in https://arxiv.org/pdf/1708.00853.pdf

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Creates a residual bock that preforms downsampling
        :param in_channels:  The number of channels fed into the block
        :param out_channels: The number of channels fed out of the block
        :param kernel_size:  The size of the kernel used for the Convolution
        """

        super(DBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,
                      padding=math.ceil((self.kernel_size - 1) / 2), stride=2),
            nn.Dropout(),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        :param x: The input signal
        :return:  The output of the block
        """

        res = self.block1(x)
        return res


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Creates a residual bock that preforms upsampling
        :param in_channels:  The number of channels fed into the block
        :param out_channels: The number of channels fed out of the block
        :param kernel_size:  The size of the kernel used for the Convolution
        """

        super(UBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,
                      padding=math.ceil((self.kernel_size - 1) / 2)),
            nn.Dropout(),
            nn.ReLU(),
            PixelShuffle1d(2)
        )
        # Conv maps (-,F,d) -> (-,F/2,d)
        # Subpixel maps (-,F/2,d) -> (-,F/4,2d)
        # concat with Dblock leads to (-,F/2,2d)

    def forward(self, x):
        """
        :param x: The input signal
        :return:  The output of the block
        """

        res = self.block1(x)
        return res


class AudioUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_resblocks=8,
                 upscale_factor=2, maxfiltersize=512, minfilterlength=9):
        """
        Builds an AudioUNet
        :param in_channels:    Number of input channels
        :param out_channels:   Number of output channels
        :param num_resblocks:  Number of downsample blocks in the network. Also the number of upsample blocks.
        :param upscale_factor: The amount the signal is unscaled by
        """

        super(AudioUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resblocks = num_resblocks
        self.maxfiltersize = maxfiltersize
        self.minfilterlength = minfilterlength
        self.upscale_factor = upscale_factor
        # I have swapped max and min from the paper

        self.n_filters = [int(min(2 ** (6 + b), self.maxfiltersize)) for b in range(1, self.num_resblocks + 1)]
        self.n_filtersizes = [int(max(2 ** (7 - b) + 1, self.minfilterlength)) for b in
                              range(1, self.num_resblocks + 1)]
        self.n_filters_changed = [1]
        self.downsampling_l = []
        self.n_filters_changed.extend(self.n_filters)

        # Creates the downsmapling blocks
        downsample_blocks = nn.ModuleList()
        for _, nfi, fs in zip(range(self.num_resblocks), range(len(self.n_filters)), self.n_filtersizes):
            downsample_blocks.append(DBlock(self.n_filters_changed[nfi], self.n_filters_changed[nfi + 1], fs))
        self.downsample_blocks = downsample_blocks

        # Creates the bottleneck block
        bottleneck_block = nn.Sequential(
            nn.Conv1d(self.n_filters_changed[-2], self.n_filters_changed[-1], self.n_filtersizes[-1],
                      padding=math.ceil((self.n_filtersizes[-1] - 1) / 2), stride=2),
            nn.Dropout(), nn.LeakyReLU(0.2))
        self.bottleneck_block = bottleneck_block

        # Creates the upsmapling blocks
        Ublock = nn.ModuleList()
        prevdim = self.n_filters_changed.copy()
        for i in range(self.num_resblocks):
            prevdim[i] = 2 * self.n_filters_changed[i + 1]

        for layer, nfi, fs in zip(range(self.num_resblocks), range(len(self.n_filters)), self.n_filtersizes):
            Ublock.append(UBlock(prevdim[layer + 1], 2 * self.n_filters_changed[nfi + 1], fs))
        self.Ublock = Ublock

        # Creates the final block
        Fblock = nn.Sequential(nn.Conv1d(prevdim[0], 2, minfilterlength, padding=math.ceil((minfilterlength - 1) / 2)),
                               PixelShuffle1d(2))
        self.Fblock = Fblock

    def forward(self, X):
        """
        :param X: The input signal
        :return:  The output of the network
        """
        x = [0 for _ in range(self.num_resblocks + 1)]
        x[0] = X

        for i in range(self.num_resblocks):
            x[i + 1] = self.downsample_blocks[i](x[i])

        y = [0 for _ in range(self.num_resblocks + 1)]
        y[0] = self.bottleneck_block(x[-1])

        for i in range(self.num_resblocks):
            temp = self.Ublock[self.num_resblocks - 1 - i](y[i])

            # crop the signal to the correct dimensions
            _, _, pr = x[self.num_resblocks - i].size()
            _, _, cu = temp.size()
            target = F.pad(input=x[self.num_resblocks - i], pad=(0, cu - pr, 0, 0, 0, 0), mode='constant', value=0)

            # Concatenate the downsampled layer with the output of the upsampled one
            y[i + 1] = torch.cat((temp, target), 1)

        # crop the output to be of the same size for addition
        _, _, las = X.size()
        fin = self.Fblock(y[-1])
        fin = fin.narrow(2, 0, las)

        # Add the global output to the global input.
        fin = fin + X
        fin[fin < 0] = 0

        # the bellow code was commented out to allow bitrate upsampling too
        # fin[fin > 2 ** 16] = 0
        # las = torch.round(fin)
        # las = torch.mul(las, 2 ** -15)
        return fin


class DBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Creates a residual bock that preforms downsampling
        :param in_channels:  The number of channels fed into the block
        :param out_channels: The number of channels fed out of the block
        :param kernel_size:  The size of the kernel used for the Convolution
        """

        super(DBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,
                      padding=math.ceil((self.kernel_size - 1) / 2), stride=2),

            nn.Dropout(),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        :param x: The input signal
        :return:  The output of the block
        """

        res = self.block1(x)
        return res


class UBlock2(nn.Module):  # This should take two inputs
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        Creates a residual bock that preforms upsampling
        :param in_channels:  The number of channels fed into the block
        :param out_channels: The number of channels fed out of the block
        :param kernel_size:  The size of the kernel used for the Convolution
        """

        super(UBlock2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.block1 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size,
                      padding=math.ceil((self.kernel_size - 1) / 2)),
            nn.Dropout(),
            nn.ReLU(),
            PhaseShuffle(20),
            PixelShuffle1d(2),
            # PhaseShuffle_Full(),
        )
        # Conv maps (-,F,d) -> (-,F/2,d)
        # Subpixel maps (-,F/2,d) -> (-,F/4,2d)
        # concat with Dblock leads to (-,F/2,2d)

    def forward(self, x):
        """
        :param x: The input signal
        :return:  The output of the block
        """

        res = self.block1(x)
        return res


class AudioUNet_shuffle(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_resblocks=8,
                 upscale_factor=2, maxfiltersize=512, minfilterlength=9):
        """
        Builds an AudioUNet
        :param in_channels:    Number of input channels
        :param out_channels:   Number of output channels
        :param num_resblocks:  Number of downsample blocks in the network. Also the number of upsample blocks.
        :param upscale_factor: The amount the signal is unscaled by
        """

        super(AudioUNet_shuffle, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_resblocks = num_resblocks
        self.maxfiltersize = maxfiltersize
        self.minfilterlength = minfilterlength
        self.upscale_factor = upscale_factor
        # I have swapped max and min from the paper

        self.n_filters = [int(min(2 ** (6 + b), self.maxfiltersize)) for b in range(1, self.num_resblocks + 1)]
        self.n_filtersizes = [int(max(2 ** (7 - b) + 1, self.minfilterlength)) for b in
                              range(1, self.num_resblocks + 1)]
        self.n_filters_changed = [1]
        self.downsampling_l = []
        self.n_filters_changed.extend(self.n_filters)

        # Creates the downsmapling blocks
        downsample_blocks = nn.ModuleList()
        for _, nfi, fs in zip(range(self.num_resblocks), range(len(self.n_filters)), self.n_filtersizes):
            downsample_blocks.append(DBlock2(self.n_filters_changed[nfi], self.n_filters_changed[nfi + 1], fs))
        self.downsample_blocks = downsample_blocks

        # Creates the bottleneck block
        bottleneck_block = nn.Sequential(
            nn.Conv1d(self.n_filters_changed[-2], self.n_filters_changed[-1], self.n_filtersizes[-1],
                      padding=math.ceil((self.n_filtersizes[-1] - 1) / 2), stride=2),
            nn.Dropout(), nn.LeakyReLU(0.2))
        self.bottleneck_block = bottleneck_block

        # Creates the upsmapling blocks
        Ublock = nn.ModuleList()
        prevdim = self.n_filters_changed.copy()
        for i in range(self.num_resblocks):
            prevdim[i] = 2 * self.n_filters_changed[i + 1]  # + self.n_filters_changed[i]
        # prevdim[0] = self.n_filters_changed[-1]
        for layer, nfi, fs in zip(range(self.num_resblocks), range(len(self.n_filters)), self.n_filtersizes):
            Ublock.append(UBlock2(prevdim[layer + 1], 2 * self.n_filters_changed[nfi + 1], fs))
        self.Ublock = Ublock

        # Creates the final block
        Fblock = nn.Sequential(nn.Conv1d(prevdim[0], 2, minfilterlength, padding=math.ceil((minfilterlength - 1) / 2)),
                               # PhaseShuffle_Full(),
                               PhaseShuffle(20),
                               PixelShuffle1d(2))
        self.Fblock = Fblock

    def forward(self, X):
        """
        :param X: The input signal
        :return:  The output of the network
        """

        x = [0 for _ in range(self.num_resblocks + 1)]
        x[0] = X

        for i in range(self.num_resblocks):
            x[i + 1] = self.downsample_blocks[i](x[i])

        y = [0 for _ in range(self.num_resblocks + 1)]
        y[0] = self.bottleneck_block(x[-1])

        for i in range(self.num_resblocks):
            temp = self.Ublock[self.num_resblocks - 1 - i](y[i])

            # crop the signal to the correct dimensions
            _, _, pr = x[self.num_resblocks - i].size()
            _, _, cu = temp.size()
            target = F.pad(input=x[self.num_resblocks - i], pad=(0, cu - pr, 0, 0, 0, 0), mode='constant', value=0)

            # Concatenate the downsampled layer with the output of the upsampled one
            y[i + 1] = torch.cat((temp, target), 1)

        # crop the output to be of the same size for addition
        _, _, las = X.size()
        fin = self.Fblock(y[-1])
        fin = fin.narrow(2, 0, las)

        # Add the global output to the global input.
        fin = fin + X
        fin[fin < 0] = 0

        # the bellow code was commented out to allow bitrate upsampling too
        # fin[fin > 2 ** 16] = 0
        # las = torch.round(fin)
        # las = torch.mul(las, 2 ** -15)

        return fin
