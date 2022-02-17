import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

from Training_Functions.Evaluation_Functions import SNR, LSD


def summarise(inp, pred, real, scale, name, out, splinedinp, from_tensor=False, spline=False):
    """
    This function is used to summarise the training process
    :param inp:         The low sample rate input signal
    :param pred:        The predicated signal
    :param real:        The real signal
    :param scale:       The scale factor to upscale by
    :param name:        The name of the file to be output
    :param out:         The output directory for figures
    :param splinedinp:  The spline interpolated value
    :param from_tensor: This says weather or not the siganl is loaded on tensor
    :param spline:      The says if the spline should be output too
    :return: The LSD and SNR loss of the predicted signal
    """

    viridisBig = cm.get_cmap('plasma')
    newcmp = ListedColormap(viridisBig(np.linspace(-2, 1.2, 512)))
    # fig = plt.figure(figsize=(20, 5))
    if from_tensor:
        pred_data = pred[0][0].sub(1).cpu().detach().numpy()
    else:
        pred_data = pred[0][0].sub(1).cpu()
    inp_data = inp[0][0].sub(1).cpu()
    real_data = real[0][0].sub(1).cpu()

    # print(pred[0][0].cpu().detach().numpy())
    if spline:
        plot_number = 4
        splinedinp = splinedinp[0][0].sub(1).cpu()
        SNRVAL = SNR(splinedinp, real_data)
        LSDVAL = LSD(splinedinp, real_data)
    else:
        plot_number = 3
    SNRVAL1 = SNR(pred_data, real_data)
    LSDVAL1 = LSD(pred_data, real_data)
    # try:
    #     plot_index = 1
    #     subplot = fig.add_subplot(scale, plot_number, plot_index)
    #     plot_index += 1
    #     plt.specgram(inp_data, Fs=12000 // scale)
    #     subplot.title.set_text('Low-Resolution, ' + 'Scale ' + str(scale))
    #     sf.write(os.path.join(out, 'Low-Resolution, ' + 'Scale ' + str(scale) + name + ".flac"), inp_data, 12000 // scale)
    #
    #     subplot = fig.add_subplot(1, plot_number, plot_index)
    #     plot_index += 1
    #     plt.specgram(pred_data, Fs=12000, NFFT=256 * scale)
    #
    #     subplot.title.set_text(
    #         'Predicted, SNR = ' + "{:.2f}".format(SNRVAL1) + '& LSD= ' + "{:.2f}".format(
    #             LSDVAL1))
    #     sf.write(os.path.join(out, 'Predicted, ' + 'Scale ' + str(scale) + name + ".flac"), pred_data, 12000)
    #     if spline:
    #         subplot = fig.add_subplot(1, plot_number, plot_index)
    #         plot_index += 1
    #         plt.specgram(splinedinp, Fs=12000, NFFT=256 * scale)
    #
    #         subplot.title.set_text(
    #             'Splined, SNR = ' + "{:.2f}".format(SNRVAL) + '& LSD= ' + "{:.2f}".format(
    #                 LSDVAL))
    #         sf.write(os.path.join(out, 'Splined, ' + 'Scale ' + str(scale) + name + ".flac"), splinedinp, 12000)
    #
    #     subplot = fig.add_subplot(1, plot_number, plot_index)
    #     plt.specgram(real_data, Fs=12000, NFFT=256 * scale)
    #     subplot.title.set_text('High-Resolution, ' + 'Scale ' + str(scale))
    #     sf.write(os.path.join(out, 'High-Resolution, ' + 'Scale ' + str(scale) + name + ".flac"), real_data, 12000)
    #     # plt.show()
    #     plt.savefig(os.path.join(out, name + '.png'))
    # except:
    #     print('rip')
    # plt.close('all')
    return (SNRVAL1, LSDVAL1)


def plot_loss(Gloss, Dloss, GScore, name, out):
    """
    This function was used to plot the loss Training_Functions independitly of weights and biases
    :param Gloss:  The generator loss
    :param Dloss:  The discriminator loss
    :param GScore: The generator score
    :param name:   The name of the method+epoch
    :param out:    The output file
    """

    return 0
    # fig = plt.figure(figsize=(20, 5))
    # subplot = fig.add_subplot(1, 3, 1)
    # plt.plot(Gloss)
    # subplot.title.set_text("Gloss")
    # subplot = fig.add_subplot(1, 3, 2)
    # plt.plot(Dloss)
    # subplot.title.set_text("Dloss")
    # subplot = fig.add_subplot(1, 3, 3)
    # plt.plot(GScore)
    # subplot.title.set_text("GScore (proportion correctly identified by the Discriminator)")
    # plt.savefig(os.path.join(out, name + '.png'))
    # plt.close('all')
