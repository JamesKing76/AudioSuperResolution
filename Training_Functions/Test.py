import torch
import wandb

from Processing.Process import process_batch
from Training_Functions.Evaluation_Functions import SNR, LSD
from Training_Functions.Interpolation_NoML import spline_interpolation


def TestModel(model, test_loader, upsample_factor, method, epoch, device):
    """
    This function tests the current model against the test set
    :param model:           The model in its current form
    :param test_loader:     The dynamic data loader
    :param upsample_factor: The factor the model is upsampling by
    :param method:          The architecture type the model is using
    :param epoch:           The epoch this test is being run in
    :param device:          The device the model is being trained on
    """

    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets, train_scale = process_batch((inputs, targets), upsample_factor)

        if method == "AudioUNet" or method == "SimpleGAN":
            N, C, V = inputs.size()
            new_inputs = []

            for j in range(N):
                new_input = spline_interpolation(train_scale, inputs[j])
                new_inputs.append(new_input)
            old_inputs = inputs.clone()
            inputs = torch.stack(new_inputs)

        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs).to(device)
        pred_data = predictions[0][0].sub(1).cpu().detach().numpy()
        real_data = targets[0][0].sub(1).cpu()

        # Calculates and logs the evaluation metrics
        SNRVAL1 = SNR(pred_data, real_data)
        LSDVAL1 = LSD(pred_data, real_data)
        wandb.log({"TESTSNR": SNRVAL1})
        wandb.log({"TESTLSD": LSDVAL1})

        # sf.write(os.path.join(
        #     '.' + os.sep + 'Data'+os.sep+'Results', 'Predict ' + 'Scale ' + str(scale) + method + 'E' + str(epoch) + 'F' + str(
        #         i) + ".flac"), pred_data, 12000)
