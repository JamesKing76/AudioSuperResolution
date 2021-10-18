import argparse

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from tqdm import tqdm_notebook

from Processing.Load_Data import load_data
from Processing.Process import process_batch
from Models.GAN_Discs import Discriminator

# This file was used to help determine if the discriminator was good at its job


scale = 2

parser = argparse.ArgumentParser()

parser.add_argument("-T", "--train_data_path", default=None, type=str, help="path of taining set")
parser.add_argument("-V", "--val_data_path", default=None, type=str, help="path of evaluation set")

args = parser.parse_args()

train_dir = args.train_data_path
test_dir = args.val_data_path


def train(model, train_loader, epoch, optimiser, losses, device, loss_function):
    """
    This function is used to train the discriminator over a noisy input
    :param model:         The model to be trained
    :param train_loader:  The dynamic loader for the samples
    :param epoch:         Epoch this iteration is train
    :param optimiser:     Epoch this iteration is train
    :param losses:        The previous losses calculated
    :param device:        The device the training is being run on
    :param loss_function: The loss function being used to train the model
    """

    tk1 = tqdm_notebook(enumerate(train_loader), total=len(train_loader), leave=False)

    for i, (inputs, targets) in tk1:
        inputs, targets, train_scale = process_batch((inputs, targets), 1)
        valid = Variable(Tensor(inputs.size(0)).fill_(1.0).to(device), requires_grad=False)
        fake = Variable(Tensor(inputs.size(0)).fill_(0.0).to(device), requires_grad=False)
        inputs = targets.to(device)
        targets = targets.to(device) + torch.normal(0, 0.2, size=inputs.size()).to(device)
        real_guess = model(inputs)
        fake_guess = model(targets)
        real_loss = loss_function(real_guess, valid)
        fake_loss = loss_function(fake_guess, fake)
        loss = (real_loss + fake_loss) / 2
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if i % 1 == 0:
            print("R:", real_loss.item(), "F:", fake_loss.item(), "T:", loss.item(), "Real Score:",
                  real_guess.mean().item(), "Fake Score", fake_guess.mean().item())
        losses.append(loss.item())

    print("Epoch " + str(epoch) + " Complete")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using: ", device)

train_loader, test_loader = load_data(train_dir, test_dir)

num_epochs = 10
best_model = [None]
best_SNR = [0.0]
best_LSD = [0.0]

losses = []
LEARNING_RATE = 1e-4
GAMMA = 0.8

model = Discriminator(in_channels=1).to(device)
loss = nn.BCELoss()
criterion = nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=GAMMA)
for epoch in range(num_epochs):
    train(model, train_loader, epoch, optimiser, losses, device, criterion)
    scheduler.step()
