import os
import random
import shutil


# This file is used to split the data into train, test and valuation set.

def Single_Split(dir_to_search):
    """
    Splits the dataset into three parts with 88:6:6 trin:test:valuation
    :param dir_to_search: The directory to search
    :return: The names of the files in each of the sets
    """

    DataSet = []

    for subdirectory, directory, files in os.walk(dir_to_search):
        for file in files:
            if file.endswith(".flac"):
                DataSet.append(tuple([os.path.join(subdirectory, file), file]))

    train_size = int(0.88 * len(DataSet))
    test_size = int(0.06 * len(DataSet))
    random.shuffle(DataSet)
    print(DataSet)
    train_dataset = DataSet[0:train_size]
    test_dataset = DataSet[train_size:train_size + test_size]
    validation_dataset = DataSet[train_size + test_size:]

    return train_dataset, test_dataset, validation_dataset


vls = 'C:\\Users\\Student\\Desktop\\Data\\wav48_silence_trimmed\\p228\\'
train_dataset, test_dataset, validation_dataset = Single_Split(vls)
print(len(train_dataset))
print(len(test_dataset))
print(len(validation_dataset))

for test in train_dataset:
    shutil.copy(test[0], os.path.join('..' + os.sep + 'Data' + os.sep + 'TRAIN', test[1]))

for test in test_dataset:
    shutil.copy(test[0], os.path.join('..' + os.sep + 'Data' + os.sep + 'TEST', test[1]))

for test in validation_dataset:
    shutil.copy(test[0], os.path.join('..' + os.sep + 'Data' + os.sep + 'VAL', test[1]))
