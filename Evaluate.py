import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap

from Processing.Filter import butter_lowpass_filter
from Models.AudioEDSR import AudioEDSR
from Models.AudioUNet import AudioUNet, AudioUNet_shuffle
from Training_Functions.Evaluation_Functions import LSD, SNR
from Training_Functions.Interpolation_NoML import spline_interpolation, linear_interpolation, flat_interpolation

# This file is used for evaluating the models

parser = argparse.ArgumentParser()
parser.add_argument("-M", "--model", default="UNET", type=str, help="EDSR|UNET|GAN")
parser.add_argument("-S", "--scale", default=2, type=str, help="The amount the model upsamples a signal by")
parser.add_argument("-P", "--modeldetail", default="", type=str, help="The postfix to the model in the file name. e.g."
                                                                      " 4EDSRL2 would have modeldetail ' L2' ")
parser.add_argument("-L", "--linear", default=False, type=bool, help="load linear data? false for deep learning models")
parser.add_argument("--single", default=False, type=bool, help="Test on single file?")
parser.add_argument("-F", "--file", default="", type=str, help="The file for single file testing")
parser.add_argument("-D", "--dir", default="." + os.path.sep + "Data" + os.path.sep + "TEST", type=str,
                    help="The directory for for multi file testing")

args = parser.parse_args()

method = args.method
scale = args.scale
ext = args.modeldetail
Linear = args.linear
single = args.single
file = args.file
wave_dir = args.dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
viridisBig = cm.get_cmap('plasma')
newcmp = ListedColormap(viridisBig(np.linspace(-2, 1.2, 512)))
#
# newcmp = ListedColormap(viridisBig(np.linspace(0, 1, 512)))
if not Linear:
    if method == "EDSR":
        model = AudioEDSR(in_channels=1, out_channels=1, num_features=128, num_resblocks=32,
                          res_scale=0.1,
                          upscale_factor=scale, kernel_size=3).to(device)

    elif method == "UNET" or method == "GAN":
        model = AudioUNet(in_channels=1, out_channels=1, num_resblocks=4, upscale_factor=scale,
                          minfilterlength=9, maxfiltersize=512).to(device)
    else:
        model = AudioUNet_shuffle(in_channels=1, out_channels=1, num_resblocks=4, upscale_factor=scale,
                                  minfilterlength=9, maxfiltersize=512).to(device)
    model.load_state_dict(
        torch.load(os.path.join("." + os.sep + "Data", str(scale) + method + ext + ".t7"),
                   map_location=torch.device('cpu')))
    # wandb.watch(model)
    print(model)
    model.eval()
    LSD_list = []
    SNR_list = []

    if single:
        # file = '.\\Data\\Test\\p228_045_mic1.flac'
        # file = "." + os.path.sep + "Data" + os.path.sep + "zeros.flac"

        # The downsamples the file
        inp = sf.read(file)[0]
        inp_filted = butter_lowpass_filter(inp, 48000 // (2 * 4), 48000).astype('float32')
        inp_filted_hr = inp_filted[::4]
        inp_filted_hr = inp_filted_hr[:len(inp_filted_hr) - (len(inp_filted_hr) % scale)]
        inp_filted_lr = torch.from_numpy( butter_lowpass_filter(
            inp_filted_hr, 12000 // (2 * scale), 12000).astype('float32'))
        inp_filted_lr = inp_filted_lr.clone()[0:len(inp_filted_lr) - (len(inp_filted_lr) % scale):scale]

        # predicts interpolates /super resolves the signal
        pred_data_spline = spline_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
        if method == "EDSR":
            pred_data = model(inp_filted_lr.detach().clone().
                              add(1).unsqueeze(0).unsqueeze(0)).sub(1).cpu().detach().numpy()[0][0]
        else:
            pred_data = model(pred_data_spline.detach().clone().unsqueeze(0)).sub(1).cpu().detach().numpy()[0][0]

        pred_data_spline = pred_data_spline.sub(1).cpu().detach().numpy()[0]
        print(pred_data_spline.shape)
        print(inp_filted_hr.shape)
        # inp_filted_lr = inp_filted_lr.sub(1)
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "PRED.flac"), pred_data, 12000)
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "HIGH.flac"), inp_filted_hr, 12000)
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "LOW.flac"), inp_filted_lr, 12000 // scale)
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "SPLINE.flac"), pred_data_spline, 12000)

        fig = plt.figure(figsize=(30, 5))
        subplot = fig.add_subplot(scale, 5, 1)
        plt.specgram(inp_filted_lr, Fs=12000 // scale, cmap=newcmp)
        subplot = fig.add_subplot(1, 5, 2)
        plt.specgram(pred_data, Fs=12000, cmap=newcmp, NFFT=256 * 2)
        subplot = fig.add_subplot(1, 5, 3)
        plt.specgram(pred_data_spline, Fs=12000, cmap=newcmp, NFFT=256 * 2)
        subplot = fig.add_subplot(1, 5, 4)
        plt.specgram(inp_filted_hr, Fs=12000, cmap=newcmp, NFFT=256 * 2)
        subplot = fig.add_subplot(1, 5, 5)
        plt.colorbar(label="ok", orientation="vertical")
        plt.show()
        print("Discrepancy with testing results")
        print("ModelLSD: ", LSD(pred_data, inp_filted_hr))
        print("SplineLSD: ", LSD(pred_data_spline, inp_filted_hr))

        print("ModelSNR: ", SNR(pred_data, inp_filted_hr))
        print("SplineSNR: ", SNR(pred_data_spline, inp_filted_hr))

        # print(max(pred_data), "<->", min(pred_data))
        # print(max(inp_filted_hr), "<->", min(inp_filted_hr))
        # print(max(inp_filted_lr), "<->", min(inp_filted_lr))
        # print(max(pred_data_spline), "<->", min(pred_data_spline))
    else:
        # file = '.\\Data\\Test\\p228_045_mic1.flac'
        # wave_dir = "." + os.path.sep + "Data" + os.path.sep + "TEST"

        files = [os.path.join(wave_dir, x) for x in os.listdir(wave_dir)]
        data = []
        for file in files:
            inp = sf.read(file)[0]

            inp_filted = butter_lowpass_filter(inp, 48000 // (2 * 4), 48000).astype('float32')
            inp_filted_hr = inp_filted[::4]
            inp_filted_hr = inp_filted_hr[:len(inp_filted_hr) - (len(inp_filted_hr) % scale)]
            print(len(inp))
            print(len(inp_filted))
            inp_filted_lr = torch.from_numpy(
                butter_lowpass_filter(inp_filted_hr, 12000 // (2 * scale), 12000).astype('float32'))
            inp_filted_lr = inp_filted_lr.clone()[0:len(inp_filted_lr) - (len(inp_filted_lr) % scale):scale]
            pred_data_spline = spline_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
            if method == "EDSR":
                pred_data = model(inp_filted_lr.detach().clone().add(1).
                                  unsqueeze(0).unsqueeze(0)).sub(1).cpu().detach().numpy()[0][0]
            else:
                pred_data = model(pred_data_spline.detach().clone().unsqueeze(0)).sub(1).cpu().detach().numpy()[0][0]
            pred_data_spline = pred_data_spline.sub(1).cpu().detach().numpy()[0]
            print(pred_data_spline.shape)
            print(inp_filted_hr.shape)
            # inp_filted_lr = inp_filted_lr.sub(1)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "PRED.flac"), pred_data, 12000)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "HIGH.flac"), inp_filted_hr, 12000)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "LOW.flac"), inp_filted_lr, 12000 // scale)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "Spline.flac"), pred_data_spline, 12000)

            fig = plt.figure(figsize=(30, 5))
            subplot = fig.add_subplot(scale, 5, 1)
            plt.specgram(inp_filted_lr, Fs=12000 // scale, cmap=newcmp)
            subplot = fig.add_subplot(1, 5, 2)
            plt.specgram(pred_data, Fs=12000, cmap=newcmp, NFFT=256 * scale)
            subplot = fig.add_subplot(1, 5, 3)
            plt.specgram(pred_data_spline, Fs=12000, cmap=newcmp, NFFT=256 * scale)
            subplot = fig.add_subplot(1, 5, 4)
            plt.specgram(inp_filted_hr, Fs=12000, cmap=newcmp, NFFT=256 * scale)
            subplot = fig.add_subplot(1, 5, 5)
            plt.colorbar(label="ok", orientation="vertical")
            plt.show()

            # LSD_list.append(LSD(pred_data, inp_filted_hr))
            # SNR_list.append(SNR(pred_data, inp_filted_hr))
            LSD_list.append(LSD(pred_data, inp_filted_hr))
            SNR_list.append(SNR(pred_data, inp_filted_hr))
            print("Discrepancy with testing results")
            print("ModelLSD: ", LSD(pred_data, inp_filted_hr))
            print("SplineLSD: ", LSD(pred_data_spline, inp_filted_hr))

            print("ModelSNR: ", SNR(pred_data, inp_filted_hr))
            print("SplineSNR: ", SNR(pred_data_spline, inp_filted_hr))

        # print(max(pred_data), "<->", min(pred_data))
        # print(max(inp_filted_hr), "<->", min(inp_filted_hr))
        # print(max(inp_filted_lr), "<->", min(inp_filted_lr))
        # print(max(pred_data_spline), "<->", min(pred_data_spline))
        print("meanLSD: ", np.mean(LSD_list))
        print("varLSD: ", np.var(LSD_list))
        print("meanSNR: ", np.mean(SNR_list))
        print("varSNR: ", np.var(SNR_list))
        # file = '.\\Data\\Test\\p228_045_mic1.flac'
        # wave_dir = "." + os.path.sep + "Data" + os.path.sep + "TEST"

        files = [os.path.join(wave_dir, x) for x in os.listdir(wave_dir)]
        data = []
        for file in files:
            inp = sf.read(file)[0]

            inp_filted = butter_lowpass_filter(inp, 48000 // (2 * 4), 48000).astype('float32')
            inp_filted_hr = inp_filted[::4]
            inp_filted_hr = inp_filted_hr[:len(inp_filted_hr) - (len(inp_filted_hr) % scale)]
            print(len(inp))
            print(len(inp_filted))
            inp_filted_lr = torch.from_numpy(
                butter_lowpass_filter(inp_filted_hr, 12000 // (2 * scale), 12000).astype('float32'))
            inp_filted_lr = inp_filted_lr.clone()[0:len(inp_filted_lr) - (len(inp_filted_lr) % scale):scale]
            pred_data_spline = spline_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)

            if method == "EDSR":
                pred_data = \
                    model(inp_filted_lr.detach().clone().add(1).unsqueeze(0).unsqueeze(0)).sub(
                        1).cpu().detach().numpy()[0][
                        0]
            else:
                pred_data = model(pred_data_spline.detach().clone().unsqueeze(0)).sub(1).cpu().detach().numpy()[0][0]
            pred_data_spline = pred_data_spline.sub(1).cpu().detach().numpy()[0]
            print(pred_data_spline.shape)
            print(inp_filted_hr.shape)

            # inp_filted_lr = inp_filted_lr.sub(1)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "PRED.flac"), pred_data, 12000)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "HIGH.flac"), inp_filted_hr, 12000)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "LOW.flac"), inp_filted_lr, 12000 // scale)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "SLINE.flac"), pred_data_spline, 12000)

            fig = plt.figure(figsize=(30, 5))
            subplot = fig.add_subplot(scale, 5, 1)
            plt.specgram(inp_filted_lr, Fs=12000 // scale, cmap=newcmp)
            subplot = fig.add_subplot(1, 5, 2)
            plt.specgram(pred_data, Fs=12000, cmap=newcmp, NFFT=256 * scale)
            subplot = fig.add_subplot(1, 5, 3)
            plt.specgram(pred_data_spline, Fs=12000, cmap=newcmp, NFFT=256 * scale)
            subplot = fig.add_subplot(1, 5, 4)
            plt.specgram(inp_filted_hr, Fs=12000, cmap=newcmp, NFFT=256 * scale)
            subplot = fig.add_subplot(1, 5, 5)
            plt.colorbar(label="ok", orientation="vertical")
            plt.show()

            # LSD_list.append(LSD(pred_data, inp_filted_hr))
            # SNR_list.append(SNR(pred_data, inp_filted_hr))
            LSD_list.append(LSD(pred_data, inp_filted_hr))
            SNR_list.append(SNR(pred_data, inp_filted_hr))

            print("Discrepancy with testing results")
            print("ModelLSD: ", LSD(pred_data, inp_filted_hr))
            print("SplineLSD: ", LSD(pred_data_spline, inp_filted_hr))

            print("ModelSNR: ", SNR(pred_data, inp_filted_hr))
            print("SplineSNR: ", SNR(pred_data_spline, inp_filted_hr))
            plt.close()
        # print(max(pred_data), "<->", min(pred_data))
        # print(max(inp_filted_hr), "<->", min(inp_filted_hr))
        # print(max(inp_filted_lr), "<->", min(inp_filted_lr))
        # print(max(pred_data_spline), "<->", min(pred_data_spline))
        print("meanLSD: ", np.mean(LSD_list))
        print("varLSD: ", np.var(LSD_list))
        print("meanSNR: ", np.mean(SNR_list))
        print("varSNR: ", np.var(SNR_list))

if Linear:
    SLSD_list = []
    FLSD_list = []
    LLSD_list = []
    SSNR_list = []
    FSNR_list = []
    LSNR_list = []
    # file = '.\\Data\\Test\\p228_045_mic1.flac'
    # wave_dir = "." + os.path.sep + "Data" + os.path.sep + "TEST"
    if single:

        inp = sf.read(file)[0]
        inp_filted = butter_lowpass_filter(inp, 48000 // (2 * 4), 48000).astype('float32')
        inp_filted_hr = inp_filted[::4]
        inp_filted_hr = inp_filted_hr[:len(inp_filted_hr) - (len(inp_filted_hr) % scale)]
        print(len(inp))
        print(len(inp_filted))
        inp_filted_lr = torch.from_numpy(
            butter_lowpass_filter(inp_filted_hr, 12000 // (2 * scale), 12000).astype('float32'))
        inp_filted_lr = inp_filted_lr.clone()[0:len(inp_filted_lr) - (len(inp_filted_lr) % scale):scale]

        pred_data_spline = spline_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
        pred_data_spline = pred_data_spline.sub(1).cpu().detach().numpy()[0]
        pred_data_linear = linear_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
        pred_data_linear = pred_data_linear.sub(1).cpu().detach().numpy()[0]
        pred_data_flat = flat_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
        pred_data_flat = pred_data_flat.sub(1).cpu().detach().numpy()[0]

        print(pred_data_spline.shape)
        print(inp_filted_hr.shape)

        fig = plt.figure(figsize=(30, 5))
        subplot = fig.add_subplot(scale, 5, 1)
        plt.specgram(inp_filted_lr, Fs=12000 // scale, cmap=newcmp, noverlap=int((12000 // scale) * 0.025),
                     NFFT=int(12000 // scale * 0.05))
        subplot = fig.add_subplot(1, 5, 2)
        plt.specgram(pred_data_flat, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
        subplot = fig.add_subplot(1, 5, 3)
        plt.specgram(pred_data_linear, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
        subplot = fig.add_subplot(1, 5, 4)
        plt.specgram(pred_data_spline, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
        subplot = fig.add_subplot(1, 5, 5)
        plt.specgram(inp_filted_hr, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
        plt.show()
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "Flat.flac"), pred_data_flat, 12000)
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "Line.flac"), pred_data_linear, 12000)
        sf.write(os.path.join(
            '.' + os.sep + 'Data',
            'Predict ' + 'Scale ' + str(scale) + "SLINE.flac"), pred_data_spline, 12000)
        # LSD_list.append(LSD(pred_data, inp_filted_hr))
        # SNR_list.append(SNR(pred_data, inp_filted_hr))
        SLSD_list.append(LSD(pred_data_spline, inp_filted_hr))
        SSNR_list.append(SNR(pred_data_spline, inp_filted_hr))
        FLSD_list.append(LSD(pred_data_flat, inp_filted_hr))
        FSNR_list.append(SNR(pred_data_flat, inp_filted_hr))
        LLSD_list.append(LSD(pred_data_linear, inp_filted_hr))
        LSNR_list.append(SNR(pred_data_linear, inp_filted_hr))
    else:
        files = [os.path.join(wave_dir, x) for x in os.listdir(wave_dir)]
        data = []
        for file in files:
            inp = sf.read(file)[0]

            inp_filted = butter_lowpass_filter(inp, 48000 // (2 * 4), 48000).astype('float32')
            inp_filted_hr = inp_filted[::4]
            inp_filted_hr = inp_filted_hr[:len(inp_filted_hr) - (len(inp_filted_hr) % scale)]
            print(len(inp))
            print(len(inp_filted))
            inp_filted_lr = torch.from_numpy(
                butter_lowpass_filter(inp_filted_hr, 12000 // (2 * scale), 12000).astype('float32'))
            inp_filted_lr = inp_filted_lr.clone()[0:len(inp_filted_lr) - (len(inp_filted_lr) % scale):scale]
            pred_data_spline = spline_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
            pred_data_spline = pred_data_spline.sub(1).cpu().detach().numpy()[0]
            pred_data_linear = linear_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
            pred_data_linear = pred_data_linear.sub(1).cpu().detach().numpy()[0]
            pred_data_flat = flat_interpolation(scale, [inp_filted_lr.clone(), 0]).add(1)
            pred_data_flat = pred_data_flat.sub(1).cpu().detach().numpy()[0]
            print(pred_data_spline.shape)
            print(inp_filted_hr.shape)
            fig = plt.figure(figsize=(30, 5))
            subplot = fig.add_subplot(scale, 5, 1)
            plt.specgram(inp_filted_lr, Fs=12000 // scale, cmap=newcmp, noverlap=int((12000 // scale) * 0.025),
                         NFFT=int(12000 // scale * 0.05))
            subplot = fig.add_subplot(1, 5, 2)
            plt.specgram(pred_data_flat, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
            subplot = fig.add_subplot(1, 5, 3)
            plt.specgram(pred_data_linear, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
            subplot = fig.add_subplot(1, 5, 4)
            plt.specgram(pred_data_spline, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
            subplot = fig.add_subplot(1, 5, 5)
            plt.specgram(inp_filted_hr, Fs=12000, cmap=newcmp, noverlap=int((12000) * 0.025), NFFT=int(12000 * 0.05))
            plt.show()
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "Flat.flac"), pred_data_flat, 12000)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "Line.flac"), pred_data_linear, 12000)
            sf.write(os.path.join(
                '.' + os.sep + 'Data',
                'Predict ' + 'Scale ' + str(scale) + "SLINE.flac"), pred_data_spline, 12000)
            # LSD_list.append(LSD(pred_data, inp_filted_hr))
            # SNR_list.append(SNR(pred_data, inp_filted_hr))
            SLSD_list.append(LSD(pred_data_spline, inp_filted_hr))
            SSNR_list.append(SNR(pred_data_spline, inp_filted_hr))
            FLSD_list.append(LSD(pred_data_flat, inp_filted_hr))
            FSNR_list.append(SNR(pred_data_flat, inp_filted_hr))
            LLSD_list.append(LSD(pred_data_linear, inp_filted_hr))
            LSNR_list.append(SNR(pred_data_linear, inp_filted_hr))

    # print(max(pred_data), "<->", min(pred_data))
    # print(max(inp_filted_hr), "<->", min(inp_filted_hr))
    # print(max(inp_filted_lr), "<->", min(inp_filted_lr))
    # print(max(pred_data_spline), "<->", min(pred_data_spline))
    print("meanLSDS: ", np.mean(SLSD_list))
    print("varLSDS: ", np.var(SLSD_list))
    print("meanSNRS: ", np.mean(SSNR_list))
    print("varSNRS: ", np.var(SSNR_list))
    print("meanLSDF: ", np.mean(FLSD_list))
    print("varLSDF: ", np.var(FLSD_list))
    print("meanSNRF: ", np.mean(FSNR_list))
    print("varSNRF: ", np.var(FSNR_list))
    print("meanLSDL: ", np.mean(LLSD_list))
    print("varLSDL: ", np.var(LLSD_list))
    print("meanSNRL: ", np.mean(LSNR_list))
    print("varSNRL: ", np.var(LSNR_list))
