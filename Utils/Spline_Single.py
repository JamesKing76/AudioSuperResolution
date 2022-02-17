import os

import soundfile as sf

from Processing.Filter import butter_lowpass_filter
from Training_Functions.Interpolation_NoML import spline_interpolation

# This file was used to analyse the effect of a spline on a file

file = '...'
inp = sf.read(file)[0]
scale = 4
inp_filted = butter_lowpass_filter(inp, 48000 // (2 * scale), 48000).tolist()
inp_filted = inp[::scale]
print(len(inp))
print(len(inp_filted))
# print([butter_lowpass_filter(inp, 48000 // (2 * scale), 48000),0])
pred_data = spline_interpolation(scale, [inp_filted, 0]).cpu().detach().numpy()[0]
print(len(pred_data))
sf.write(os.path.join(
    '..' + os.sep + 'Data',
    'Predict ' + 'Scale ' + str(scale) + "Bspline.flac"), pred_data, 48000)
