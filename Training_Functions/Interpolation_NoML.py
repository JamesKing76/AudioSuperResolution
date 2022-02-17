import numpy as np
import scipy.interpolate as interp
import torch


def spline_interpolation(scale, inp):
    """
    Performs cubic spline interpolation
    :param scale: The scale to upsample the signal by
    :param inp:   The data to be upsampled
    :return:      The output tensor post spline
    """

    inp = inp[0]
    calc_new_data = interp.CubicSpline([i * scale for i in range(len(inp))], inp)
    new_data = [0] * len(inp) * scale
    new_data = np.array(calc_new_data([i for i in range(len(new_data))])).astype('float32')

    return torch.from_numpy(new_data).unsqueeze(0)


def flat_interpolation(scale, inp):
    """
    Performs flat interpolation
    :param scale: The scale to upsample the signal by
    :param inp:   The data to be upsampled
    :return: The output tensor post interpolation
    """

    inp = inp[0]
    new_data = np.zeros(len(inp) * scale)

    for i in range(scale):
        new_data[i::scale] = inp
    new_data = np.array(new_data).astype('float32')

    return torch.from_numpy(new_data).unsqueeze(0)


def linear_interpolation(scale, inp):
    """
    Performs piecewise linear interpolation (flat for final sample)
    :param scale: The scale to upsample the signal by
    :param inp:   The data to be upsampled
    :return: The output tensor post interpolation
    """

    inp = inp[0]
    new_inp = np.zeros(len(inp) + 1)
    new_inp[0:len(inp):] = inp
    new_inp[-1] = inp[-1]
    new_data = [] * len(inp) * scale

    for i in range(len(inp)):
        new_data[i * scale::] = [new_inp[i]+j*((new_inp[i+1] - new_inp[i])/scale) for j in range(scale)]
    new_data = np.array(new_data).astype('float32')

    return torch.from_numpy(new_data).unsqueeze(0)
