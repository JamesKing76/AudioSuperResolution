import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from Training_Functions.Interpolation_NoML import spline_interpolation


# An implementation of 1d pixel shuffle from: https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b

def pixel_shuffle_1d(x, upscale_factor):
    """
    Performs a pixel shuffle on the input signal
    :param x:              The input tensor to be dimension shuffled
    :param upscale_factor: The upsample factor
    :return: The shuffled tensor
    """

    batch_size, channels, steps = x.size()
    channels //= upscale_factor
    input_view = x.contiguous().view(batch_size, channels, upscale_factor, steps)
    shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()

    return shuffle_out.view(batch_size, channels, steps * upscale_factor)


class PixelShuffle1d(Module):
    """
    1D pixelshuffle module
    """
    __constants__ = ['upscale_factor']
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super(PixelShuffle1d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return pixel_shuffle_1d(input, self.upscale_factor)  # changed to use our 1d pixel shuffle function

    def extra_repr(self) -> str:
        return 'upscale_factor={}'.format(self.upscale_factor)


class Spline_up(Module):
    """
    1D spline upsample module
    """
    __constants__ = ['upscale_factor']
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super(Spline_up, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return spline_interpolation(self.upscale_factor,
                                    input.cpu().detach())  # changed to use our 1d pixel shuffle function

    def extra_repr(self) -> str:
        return 'upscale_factor={}'.format(self.upscale_factor)


class Upsampler(nn.Module):
    def __init__(self, upsample_factor, kernel_size=3, num_features=64):
        """
        Builds the upsample block for the AudioEDSR network
        :param upsample_factor: The factor to upscale the signal by
        :param num_features:    The number of features used in the upsample convolution
        """
        super(Upsampler, self).__init__()
        self.scale = upsample_factor
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.build_block()

    def build_block(self):
        """
        Creates the upsample block
        """
        m = []

        if (self.scale & (self.scale - 1)) == 0:

            for _ in range(int(math.log(self.scale, 2))):  # CHECK THIS
                m.append(nn.Conv1d(self.num_features, 2 * self.num_features, self.kernel_size, stride=1,
                                   padding=math.ceil((self.kernel_size - 1) / 2)))
                m.append(PixelShuffle1d(2))
                # m.append(PhaseShuffle(2))
                m.append(nn.LeakyReLU(0.2))

        self.block1 = nn.Sequential(*m)

    def forward(self, x):
        """
        :param x: The input signal
        :return:  The output of the block
        """

        out = self.block1(x)

        return out


# from https://github.com/rahulbhalley/wavegan.pytorch/blob/master/wavegan_1_024.py

class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    # Adapted from https://github.com/jtcramer/wavegan/
    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle


class PhaseShuffle_Full(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """

    def __init__(self):
        super(PhaseShuffle_Full, self).__init__()

    def forward(self, x):
        self.shift_factor = x.shape[2] // 2 - 1
        if self.shift_factor == 0:
            return x
        # uniform in (L, R)
        k_list = torch.Tensor(x.shape[0]).random_(0, 2 * self.shift_factor + 1) - self.shift_factor
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            k_map[k].append(idx)

        # Make a copy of x for our output
        x_shuffle = x.clone()

        # Apply shuffle to each sample
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode='reflect')
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape, x.shape)
        return x_shuffle
