import argparse

import torch

from Processing.Load_Data import load_data
from Processing.Process import process_batch
from Training_Functions.Sumerise import summarise
from Training_Functions.Interpolation_NoML import spline_interpolation, flat_interpolation, linear_interpolation

# This file is used to perform non-machine learning based interpolation.

parser = argparse.ArgumentParser()
parser.add_argument("-V", "--val_data_path", default=None, type=str, help="path of evaluation set")
parser.add_argument("-M", "--method", default="Spline", type=str, help="Spline| Linear| Flat")
parser.add_argument("-O", "--output_path", default=".", type=str, help="Output directory")
parser.add_argument("-S", "--scale", default=2, type=int, help="Upscale factor")
args = parser.parse_args()

test_dir = args.val_data_path
method = args.method
out_dir = args.output_path
scale = args.scale
_, test_loader = load_data(None, test_dir)

with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        if i == 2:
            break
        inputs, targets, test_scale = process_batch((inputs, targets),scale)
        N, C, V = inputs.size()
        predictions = []
        for j in range(N):
            if method == "Spline":
                prediction = spline_interpolation(test_scale, inputs[j])
            elif method == "Flat":
                prediction = flat_interpolation(test_scale, inputs[j])
            elif method == "Linear":
                prediction = linear_interpolation(test_scale, inputs[j])
            predictions.append(prediction)

        predictions = torch.stack(predictions)
        summarise(inputs, predictions, targets, test_scale, method + str(i), out_dir, None)
