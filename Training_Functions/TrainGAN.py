import numpy as np
import torch
import torch.autograd as autograd
import wandb
from torch import Tensor
from torch.autograd import Variable
from tqdm.autonotebook import tqdm

from Processing.Process import process_batch
from Training_Functions.Interpolation_NoML import spline_interpolation
from Training_Functions.Sumerise import summarise, plot_loss


def trainAudioUNetGAN(epoch, train_loader, method, netG, netD, device, new_loss, optimiserD, optimiserG, g_losses,
                      d_losses,
                      g_scores, out_dir, scale, discupdaterate, lambdapar):
    """
    Trains the AudioUNetGAN
    :param epoch:          Epoch this iteration is train
    :param train_loader:   The dynamic loader for the samples
    :param method:         The name of the method being used for the architecture
    :param netG:           The generator network
    :param netD:           The discriminator network
    :param device:         The device the model is training on
    :param new_loss:       The non-adversarial loss function used
    :param optimiserD:     The optimiser for the discriminator
    :param optimiserG:     The optimiser for the generator
    :param g_losses:       An array of the generator losses
    :param d_losses:       An array of the discriminator losses
    :param g_scores:       An array of the generator scores (the score the discriminator gives generated samples)
    :param out_dir:        The directory to output any summarised data to
    :param scale:          The scale of upsample problem we are solving
    :param discupdaterate: The number of times the discriminator is updated over the generator
    :param lambdapar:      The hyperparameter altering the proportion of the loss that is adversarial
    """

    Data_Loader = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for i, (inputs, targets) in Data_Loader:
        inputs, targets, train_scale = process_batch((inputs, targets), scale)
        N, C, V = inputs.size()
        new_inputs = []

        for j in range(N):
            new_input = spline_interpolation(train_scale, inputs[j])
            new_inputs.append(new_input)
        old_inputs = inputs.clone()
        inputs = torch.stack(new_inputs)
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Train the discriminator:
        d_loss = trainDiscriminator(netG, netD, optimiserD, inputs, targets, new_loss, device)

        # Train the generator:
        if i % discupdaterate == 0:
            g_loss = trainGenerator(netG, netD, optimiserG, inputs, targets, new_loss, device, lambdapar)

        fake_waves = netG(inputs.clone())
        fake_out = netD(fake_waves).mean()
        real_out = netD(targets).mean()

        if i % discupdaterate == 0:
            gen_loss = g_loss.item()
        discrim_loss = d_loss.item()
        gan_score = fake_out.item()
        Discfakescore = real_out.item()

        if i % int(len(train_loader) / 10) == 0:
            snr, lsd = summarise(old_inputs, fake_waves, targets, train_scale, method + "E" + str(epoch) + "I" + str(i),
                                 out_dir, inputs, from_tensor=True,
                                 spline=True)
            plot_loss(g_losses, d_losses, g_scores, "Losses" + method + "E" + str(epoch), out_dir)

            if gen_loss:
                wandb.log({"Generator loss": gen_loss})
            wandb.log({"Discriminator loss": discrim_loss})
            wandb.log({"proportion generator fools discriminator": gan_score})
            wandb.log({"average discriminator score on real data": Discfakescore})
            wandb.log({"SNR": snr})
            wandb.log({"LSD": lsd})

    print("Epoch " + str(epoch) + " Complete")


def trainGenerator(netG, netD, optimiserG, inputs, targets, loss_func, device, lambdapar):
    """
    This function trains the generator
    :param netG:       The generator network
    :param netD:       The discriminator network
    :param optimiserG: The optimiser for the generator
    :param inputs:     The low resolution input signal for training
    :param targets:    The high resolution target signal for training
    :param loss_func:  The non-adversarial loss function used
    :param device:     The device the model is training on
    :param lambdapar:  The hyperparameter altering the proportion of the loss that is adversarial
    :return: The generator loss
    """

    optimiserG.zero_grad()
    fake_waves = netG(inputs.clone()).to(device)
    fake_validity = netD(fake_waves.clone())
    normal_loss = loss_func(fake_waves.clone(), targets.clone())
    g_loss = normal_loss - lambdapar * torch.mean(fake_validity)
    g_loss.backward()
    optimiserG.step()
    return g_loss


def trainDiscriminator(netG, netD, optimiserD, inputs, targets, loss_func, device):
    """
    This function trains the discriminator
    :param netG:       The generator network
    :param netD:       The discriminator network
    :param optimiserD: The optimiser for the discriminator
    :param inputs:     The low resolution input signal for training
    :param targets:    The high resolution target signal for training
    :param loss_func:  The non-adversarial loss function used
    :param device:     The device the model is training on
    :return:  The discriminator loss
    """

    optimiserD.zero_grad()
    fake_waves = netG(inputs.clone()).to(device)
    real_waves = Variable(targets.type(Tensor).to(device))
    real_validity = netD(real_waves.clone())
    fake_validity = netD(fake_waves.clone())
    gradient_penalty = compute_gradient_penalty(netD, real_waves.clone().data, fake_waves.clone().data, device)
    # Adversarial loss

    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
    d_loss.backward()
    optimiserD.step()
    return d_loss


# adapted from
# https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/wgan_gp/wgan_gp.py#L119

def compute_gradient_penalty(discrim, real_samples, generated_samples, device):
    """
    This function computes the gradient penalty WGAN-GP training
    :param discrim:           The discriminator
    :param real_samples:      Samples taken from the dataset
    :param generated_samples: Samples generated from the
    :param device:            The device the model is training on
    :return: The gradient penalty score
    """

    # Random weight term for interpolation
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1))).to(device)

    # Gets the random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * generated_samples)).requires_grad_(True)
    d_interpolates = discrim(interpolates)
    fake = Variable(Tensor(real_samples.size(0)).fill_(1.0).to(device), requires_grad=False)

    # Compute the gradients
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute the gradient penalties
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
