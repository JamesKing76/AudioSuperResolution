from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, lower_bound, freq, order=8):
    """
    Performs a low pass butterworth filter
    :param data:        The data of the signal to be lowpassed
    :param lower_bound: The low cut of frequency
    :param freq:        The frequency for FFT
    :param order:       The order of the lowpass
    :return: the data for the signal post lowpass
    """
    low = lower_bound / (0.5 * freq)
    under, upper = butter(order, low, btype='lowpass')
    y = lfilter(under, upper, data)
    return y
