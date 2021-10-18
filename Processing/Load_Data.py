from torch.utils.data import DataLoader

from Processing.Process import LoadTrainingData, LoadTestData


# Load training and test datasets, based on tutorial.


def load_data(train_dir, test_dir, batch_size=1600, crop_size=2**15,):
    """
    This crates the dataloader
    :param train_dir:  The path to the training directory
    :param test_dir:   The path to the test directory
    :param batch_size: The number of samples in a single batch
    :param crop_size:  The size of the a single patch
    :return: both a dynamic test leader and dynamic train loader
    """

    if train_dir:
        train_set = LoadTrainingData(train_dir, num_patches=64, crop_size=crop_size)
        training_data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    else:
        training_data_loader = None

    test_set = LoadTestData(test_dir, crop_size=2**20)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    return training_data_loader, testing_data_loader

