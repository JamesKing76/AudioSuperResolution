import argparse
import os

import soundfile as sf
import torch
import torch.nn as nn
import wandb

from Processing.Load_Data import load_data
from Processing.Process import process_batch
from Models.AudioEDSR import AudioEDSR
from Models.AudioUNet import AudioUNet
from Models.GAN_Discs import Discriminator
from Training_Functions.Test import TestModel
from Training_Functions.TrainGAN import trainAudioUNetGAN
from Training_Functions.TrainNonGAN import train

# 1. Start a new run

wandb.init(project='AudioSuperResolutionProject', entity='audiosuperresolutionpartii')

# 2. Save model inputs and hyperparameters
config = wandb.config

# Arguments for input
parser = argparse.ArgumentParser()
parser.add_argument("-M", "--method", default="AudioUNet", type=str, help="EDSR|AudioUNet|SimpleGAN")
parser.add_argument("-B", "--blocks", default=32, type=int, help="number of residual blocks")
parser.add_argument("-F", "--features", default=256, type=int, help="number of features")
parser.add_argument("-O", "--output_path", default=".", type=str, help="Output directory")
parser.add_argument("-T", "--train_data_path", default='.' + os.sep + 'Data' + os.sep + 'TRAIN' + os.sep, type=str,
                    help="path of taining set")
parser.add_argument("-V", "--val_data_path", default='.' + os.sep + 'Data' + os.sep + 'VAL' + os.sep, type=str,
                    help="path of evaluation set")
parser.add_argument("-X", "--test_data_path", default='.' + os.sep + 'Data' + os.sep + 'TEST' + os.sep, type=str,
                    help="path of test set")
parser.add_argument("--verbose", default=False, type=bool, help="print more info?")
parser.add_argument("-S", "--scale", default=2, type=int, help="Upscale factor")
parser.add_argument("-C", "--crop", default=6000, type=int, help="Crop size")
parser.add_argument("-K", "--batchsize", default=64, type=int, help="batchsize")
parser.add_argument("--genlrate", default=1e-3, type=float, help="generator learning rate")
parser.add_argument("--disclrate", default=1e-3, type=float, help="Discriminator learning rate")
parser.add_argument("--duprate", default=1, type=int, help="Discriminator update rate per generator update")
parser.add_argument("--discblocks", default=7, type=int, help="Discriminator Block Numbers")
parser.add_argument("--b1", default=0.9, type=float, help="beta1")
parser.add_argument("--b2", default=0.999, type=float, help="beta2")
parser.add_argument("-P", "--maxflitersize", default=512, type=int, help="max fliter size")
parser.add_argument("-Q", "--minfilterlength", default=9, type=int, help="min filter length/kernel size")
parser.add_argument("--lambdapar", default=0.001, type=float, help="complex loss")
parser.add_argument("--loss", default="L1", type=str, help="L1|L2")
parser.add_argument("-E", "--epochs", default=4000, type=int, help="The number of epochs to apply the model too")

args = parser.parse_args()

method = args.method
num_resblocks = args.blocks
features = args.features
train_dir = args.train_data_path
val_dir = args.val_data_path
test_dir = args.test_data_path
out_dir = args.output_path
verbose = args.verbose
scale = args.scale
crop = args.crop
batchsize = args.batchsize
genlrate = args.genlrate
disclrate = args.disclrate
Discupdaterate = args.duprate
DiscBlocks = args.discblocks
b1 = args.b1
b2 = args.b2
maxflitersize = args.maxflitersize
minfilterlength = args.minfilterlength
lambdapar = args.lambdapar
loss_function_para = args.loss


def init_weights(model):
    """
    This function initialises the weights of a model
    :param model: The model with weights to be initialised
    """

    if isinstance(model, nn.Conv1d or nn.Linear):
        torch.nn.init.xavier_normal_(model.weight)
    # elif isinstance(m, AudioUNet.DBlock or AudioUNet.UBlock):
    #     print(str(type(m)))
    # else:
    #     print("failed" + str(type(m)))


train_loader, test_loader = load_data(train_dir, test_dir, crop_size=crop, batch_size=batchsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using: ", device)

num_epochs = 40000
best_model_SNR = None
best_model_LSD = None
best_SNR = -100
best_LSD = 0.0

losses = []

if method == "SimpleGAN":
    LEARNING_RATE_GENERATOR = genlrate
    LEARNING_RATE_DISCRIMINATOR = disclrate
    GAMMA = 1
    netG = AudioUNet(num_resblocks=num_resblocks, upscale_factor=scale, minfilterlength=minfilterlength,
                     maxfiltersize=maxflitersize).to(device)
    # netG = EDSR(in_channels=1, out_channels=1, num_features=features, bit_depth=2 ** 16, num_resblocks=num_resblocks,
    #              res_scale=0.1,
    #              upscale_factor=UPSCALE_FACTOR).to(device)
    netD = Discriminator(DiscBlocks, in_channels=1).to(device)

    g_losses = []
    d_losses = []
    g_scores = []

    optimiserG = torch.optim.Adam(netG.parameters(), lr=LEARNING_RATE_GENERATOR, betas=[b1, b2])
    optimiserD = torch.optim.Adam(netD.parameters(), lr=LEARNING_RATE_DISCRIMINATOR, betas=[b1, b2])
    schedulerG = torch.optim.lr_scheduler.StepLR(optimiserG, step_size=500, gamma=GAMMA)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimiserD, step_size=500, gamma=GAMMA)
    netG.apply(init_weights)
    netD.apply(init_weights)

    # Watch the generator and the discriminator in weights and biases
    wandb.watch(netD)
    wandb.watch(netG)

    if loss_function_para == "L1":
        new_loss = nn.L1Loss()
    else:
        new_loss = nn.MSELoss()

    # Outputs the files
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets, train_scale = process_batch((inputs, targets),scale)
        targets = targets.to(device)
        real_data = targets[0][0].sub(1).cpu()
        sf.write(os.path.join('.' + os.sep + 'Data' + os.sep + 'Results',
                              'Real ' + 'Scale ' + str(scale) + method + 'F' + str(i) + ".flac"), real_data, 12000)

    for epoch in range(num_epochs):
        if method == "SimpleGAN":
            trainAudioUNetGAN(epoch, train_loader, method, netG, netD, device, new_loss, optimiserD, optimiserG,
                              g_losses, d_losses, g_scores, out_dir, scale, Discupdaterate, lambdapar)

        TestModel(netG, test_loader, scale, method, epoch, device)

        # saves the models after evey epoch to stop a failed training making a large impact
        torch.save(netG.state_dict(), os.path.join(
            '.' + os.sep + 'Data', str(scale) + "Current" + method + ".t7"))
        torch.save(netD.state_dict(), os.path.join(
            '.' + os.sep + 'Data', str(scale) + "Current" + method + "DISC.t7"))

else:
    LEARNING_RATE = genlrate
    GAMMA = 0.8

    if method == "EDSR":
        model = AudioEDSR(in_channels=1, out_channels=1, num_features=features, num_resblocks=num_resblocks,
                          res_scale=0.1,
                          upscale_factor=scale, kernel_size=minfilterlength).to(device)

    elif method == "AudioUNet":
        model = AudioUNet(in_channels=1, out_channels=1, num_resblocks=num_resblocks, upscale_factor=scale,
                                  minfilterlength=minfilterlength, maxfiltersize=maxflitersize).to(device)

    if loss_function_para == "L1":
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=GAMMA)
    model.apply(init_weights)
    wandb.watch(model)

    for epoch in range(num_epochs):
        train(model, train_loader, epoch, optimiser, criterion, losses, out_dir, device, method, scale)
        scheduler.step()
        TestModel(model, test_loader, scale, method, epoch, device)
        # torch.save(rets[0], '.' + os.sep + 'Data' + os.sep + "Best_SNR" + ".t7")
        # torch.save(rets[1], '.' + os.sep + 'Data' + os.sep + "Best_LSD" + ".t7")
        torch.save(model.state_dict(), os.path.join(
            '.' + os.sep + 'Data', str(scale) + "Current" + method + loss_function_para + ".t7"))
