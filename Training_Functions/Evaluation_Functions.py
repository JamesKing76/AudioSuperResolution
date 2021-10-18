import math

import numpy as np
import scipy.signal as signal


# calculates the SNR of the data
def SNR(origonal_data, new_data):
    """
    Calculates the signal-to-noise ratio of a signal
    :param origonal_data: The signal to compare the to
    :param new_data:      The signal to compute the signal-to-noise ratio on
    :return:              The signal-to-noise ratio score
    """

    numer = np.linalg.norm(new_data) ** 2
    denom = np.linalg.norm(new_data - origonal_data) ** 2

    return 10 * math.log10(numer / denom)


def LSD(original_data, new_data, freq=48000):
    """
    Computes the log spectral distance of a signal
    :param original_data: The signal to compare the to
    :param new_data:      The signal to compute the signal-to-noise ratio on
    :param freq:          The frequency for the fast fourier transform
    :return:              The log spectral distance score
    """

    W, K, ori_f = signal.stft(original_data, fs=freq)
    W, K, new_f = signal.stft(new_data, fs=freq)
    finsum = 0

    for w in range(len(W)):
        subsum = 0
        for k in range(len(K)):
            interior = abs(ori_f[w][k]) ** 2 / abs(new_f[w][k]) ** 2
            assert interior >= 0

            if interior != 0:
                subsum += math.log10(interior) ** 2

        finsum += math.sqrt((1 / len(K)) * subsum)

    return (1 / len(W)) * finsum


# Code https://github.com/kuleshov/audio-super-res/issues/25
# Note this code is from the AudioUNet paper and uses log_e which I dont think is correct
#
# def LSD_from_paper(x_hr, x_pr):
#     def get_power(x):
#         S = librosa.stft(x, 2048 // 4)
#         S = np.log(np.abs(S) ** 2)
#         return S
#
#     with np.errstate(divide='ignore'):
#         S1 = get_power(np.array(x_hr))
#         S2 = get_power(np.array(x_pr))
#         lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2, axis=1)), axis=0)
#     return lsd
