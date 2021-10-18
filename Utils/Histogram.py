import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# This file was used for generating the histogram

wave_dir_dir = 'C:\\Users\\Student\\Desktop\\Data\\wav48_silence_trimmed\\'
wave_dirs = [os.path.join(wave_dir_dir, x) for x in os.listdir(wave_dir_dir)]
i = 1
maxi = 0
mini = 0
all = []
for wave_dir in wave_dirs:
    files = [os.path.join(wave_dir, x) for x in os.listdir(wave_dir)]
    data = []
    for file in files:
        data.extend(sf.read(file)[0])  # reading wave file.
    i += 1
    all.extend(data)
    print(i)
    # maxi = max(np.max(data), maxi)
    # mini = min(np.min(data), mini)
    # print(maxi)
    # print(mini)
binwidth = 0.001
plt.hist(all, bins=np.arange(-1, 1, binwidth), weights=np.ones_like(all) / len(all))
plt.show()
#     c = random.sample(list, 10000)  # reading first 500 samples from data variable with contain 200965 samples.
#     c = data
# plt.hist(c, bins='auto')  # arguments are passed to np.histogram.
# plt.title("Histogram with 'auto' bins")
# plt.show()

print(maxi)
print(mini)
