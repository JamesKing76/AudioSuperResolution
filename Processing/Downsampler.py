import os

import numpy as np
import soundfile as sf


def retrive_data_low(file, downsample_rate, n, m):
    """
    This function downsamples a file and extracts its lowpoints and crops it to be of an appropriate size
    :param file:            The audio file to be downasmpled
    :param downsample_rate: The amount we downsample the singal by
    :param n:               The amount before extracted in each set
    :param m:               The amount after extracted in each set
    :return:                An array of the set of low points
    """

    data_points, sample_rate = sf.read(file)
    new_data_points = data_points[0::downsample_rate]
    new_paded_data_points = np.pad(new_data_points, (n, m))
    data = [[]] * (len(new_data_points))

    for i in range(0, len(new_data_points)):
        data[i] = new_paded_data_points[i + 1:i + 1 + m + n:]

    return np.array(data)


def retrive_data_high(file, downsample_rate):
    """
    This function crops the file and extracts the high resolution samples
    :param file:            The audio file to be downasmpled
    :param downsample_rate: The amount we downsample the signal by
    :return:                And array of the high points
    """

    data_points, sample_rate = sf.read(file)
    modesumtemp = len(data_points) % downsample_rate
    new_paded_data_points = np.zeros(
        len(data_points) + (0 if modesumtemp == 0 else (downsample_rate - modesumtemp)))
    new_paded_data_points[0:len(data_points):] = data_points
    data = [[0 in range(downsample_rate)]] * (len(new_paded_data_points) // downsample_rate)

    for i in range(0, len(data)):
        data[i] = new_paded_data_points[downsample_rate * i:downsample_rate * i + downsample_rate:]

    return np.array(data)


def retrive_from_directory(dir_to_search, downsample_rate, n, m):
    """
    This function gets both the high and low samples from every file in a directory
    :param dir_to_search:   The directory to extract the files from
    :param downsample_rate: The amount we downsample the signal by
    :param n:               The amount before extracted in each set in the low extraction
    :param m:               The amount after extracted in each set in the low extraction
    :return:                Both the set of low points and the high points for each file in the directory
    """

    total_high = np.zeros((0, downsample_rate))
    total_low = np.zeros((0, m + n))

    for subdirectory, directory, files in os.walk(dir_to_search):
        for file in files:
            if file.endswith(".flac"):
                filepath = subdirectory + os.sep + file
                current = retrive_data(filepath, downsample_rate, n, m)
                total_low = np.concatenate((total_low, current[0]), axis=0)
                total_high = np.concatenate((total_high, current[1]), axis=0)

    return total_low, total_high


def retrive_data(file, downsample_rate, n, m):
    """
    This function retrieves data from a single file
    :param file:            The audio file to be downasmpled
    :param downsample_rate: The amount we downsample the signal by
    :param n:               The amount before extracted in each set in the low extraction
    :param m:               The amount after extracted in each set in the low extraction
    :return:
    """

    return retrive_data_low(file, downsample_rate, n, m), retrive_data_high(file, downsample_rate)
