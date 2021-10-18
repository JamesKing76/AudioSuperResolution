import torch
import wandb
from tqdm.autonotebook import tqdm

from Processing.Process import process_batch
from Training_Functions.Interpolation_NoML import spline_interpolation
from Training_Functions.Sumerise import summarise


def train(model, train_loader, epoch, optimiser, loss_function, losses, out_dir, device, method, upsample_factor):
    """
    This function is used for training the non GAN based methods.
    :param model:           The model to be trained
    :param train_loader:    The dynamic loader for the samples
    :param epoch:           Epoch this iteration is train
    :param optimiser:       The optimiser for the model
    :param loss_function:   The loss function being used to train the model
    :param losses:          The previous losses calculated
    :param out_dir:         The directory any files produced by this program are sent to
    :param device:          The device the training is being run on
    :param method:          The architecture the model is based on
    :param upsample_factor: The factor the model is being trained to upsample a signal by
    """

    tk1 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for i, (inputs, targets) in tk1:
        inputs, targets, train_scale = process_batch((inputs, targets), upsample_factor)
        if method == "AudioUNet":
            N, C, V = inputs.size()
            new_inputs = []
            for j in range(N):
                new_input = spline_interpolation(train_scale, inputs[j])
                new_inputs.append(new_input)
            old_inputs = inputs.clone()
            inputs = torch.stack(new_inputs)

        optimiser.zero_grad()

        inputs = inputs.clone().to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        predictions = predictions.to(device)

        loss = loss_function(predictions, targets)
        loss.backward()
        optimiser.step()

        # if i % int(len(train_loader)/10) == 0:
        if i % int(len(train_loader) / 10) == 0:
            # print(i)
            if method == "AudioUNet":
                vals = summarise(old_inputs, predictions, targets, train_scale,
                                 method + "E" + str(epoch) + "I" + str(i),
                                 out_dir, inputs, from_tensor=True,
                                 spline=True)

            else:
                vals = summarise(inputs, predictions, targets, train_scale, method + "E" + str(epoch) + "I" + str(i),
                                 out_dir,
                                 None,
                                 from_tensor=True)

            wandb.log({"loss": loss})
            wandb.log({"SNR": vals[0]})
            wandb.log({"LSD": vals[1]})
        losses.append(loss.item())

    print("Epoch " + str(epoch) + " Complete")
