import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# This file was used to generate empty signals

af = np.zeros(48000 * 5)
# x = np.linspace(0, 48000 * np.pi, 48000*16)
# af =  np.sin(x)
af[::1] = 0.00
sf.write(".." + os.path.sep + "Data" + os.path.sep + "zeros.flac", af, 48000)
plt.specgram(af, Fs=48000)
plt.show()
