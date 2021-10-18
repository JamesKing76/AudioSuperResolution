import os

import numpy
import soundfile as sf
import torch
from torch.utils.data import Dataset

from Processing.Filter import butter_lowpass_filter


# Based on the tutorial

class LoadTrainingData(Dataset):
    def __init__(self, wave_dir, num_patches=32, crop_size=24000):
        """
        This function initialises the dataset for training
        :param wave_dir:    The directory that the audio signals are stored in
        :param num_patches: The number of to extract per signal
        :param crop_size:   The size of each patch
        """

        super(LoadTrainingData, self).__init__()
        self.wave_filenames = [os.path.join(wave_dir, x) for x in os.listdir(wave_dir)]
        self.num_patches = num_patches
        self.crop_size = crop_size

    def __getitem__(self, index):
        """
        This function gets the pairs of input and target signals
        :param index: The index of the signal we retrieve the pair from
        :return: Input and target tensors (pre downsample)
        """

        index = index // self.num_patches
        wave = numpy.array(sf.read(self.wave_filenames[index])[0]).astype('float32')
        wave = numpy.array(wave).astype('float32')

        # Pass the signal though a bandpass filter
        wave = torch.from_numpy(butter_lowpass_filter(wave, 48000 // (2 * 4), 48000).astype('float32'))

        # Crop the signal
        start = numpy.random.randint(0, len(wave) - self.crop_size)
        inp = wave[start:start + self.crop_size:4].unsqueeze(0)
        target = inp.detach().clone()

        return inp, target

    def __len__(self):
        """
        :return: The number of samples in the training data
        """

        return len(self.wave_filenames) * self.num_patches


class LoadTestData(Dataset):
    def __init__(self, wave_dir, crop_size=24000):
        """
        This function initialises the dataset for testing
        :param wave_dir:    The directory that the audio signals are stored in
        :param crop_size:   The size of each patch
        """

        super(LoadTestData, self).__init__()
        self.wave_filenames = [os.path.join(wave_dir, x) for x in os.listdir(wave_dir)]
        self.crop_size = crop_size

    def __getitem__(self, index):
        """
        This function gets the pairs of input and target signals
        :param index: The index of the signal we retrieve the pair from
        :return: Input and target tensors (pre downsample)
        """

        wave = numpy.array(sf.read(self.wave_filenames[index])[0]).astype('float32')
        wave = numpy.array(wave).astype('float32')
        # wave = numpy.array(2**16*wave).astype('float32')
        start = len(wave) // 2
        # inp = torch.from_numpy(wave[start:start + self.crop_size:]).unsqueeze(0)
        # target = torch.from_numpy(wave[start:start + self.crop_size:]).unsqueeze(0)
        wave = torch.from_numpy(butter_lowpass_filter(wave, 48000 // (2 * 4), 48000).astype('float32'))
        inp = wave[::4].unsqueeze(0)
        target = inp.detach().clone()
        return inp, target

    def __len__(self):
        """
        :return: The number of samples in the training data
        """

        return len(self.wave_filenames)


def process_batch(batch, upsample_factor):
    """
    Processes a batch
    :param upsample_factor: The amount the file is to be upsampled by in training
    :param batch:           A batch is in the form of a tuple (inputs, targets).
    :return The inputs post lowpassfilter and donwsmapling and the cropped targets
    """

    inputs, targets = batch
    N, C, V = inputs.shape

    crop_size = V - V % upsample_factor
    new_inputs = []
    new_targets = []

    for i in range(N):
        inp = inputs[i][0]
        target = targets[i][0].add(1)
        inp = torch.from_numpy(butter_lowpass_filter(inp, 12000 // (2 * upsample_factor), 12000).astype('float32'))
        inp = inp[0:crop_size:upsample_factor].add(1)
        new_inputs.append(inp)
        new_targets.append(target[:crop_size:])

    return torch.stack(new_inputs).unsqueeze(1), torch.stack(new_targets).unsqueeze(1), upsample_factor
