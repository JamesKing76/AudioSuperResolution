import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

# This file was used to analyse the dataset

file = '...'
data = sf.read(file)[0]
hs1 = hilbert(data)
x = np.linspace(0, 2 * np.pi, 1000)
fu = np.sin(x)
hs2 = hilbert(fu)
plt.plot(np.real(hs2), np.imag(hs2), 'b')
# plt.plot(np.real(hs1), np.imag(hs1), 'r')
# omega_s1 = np.unwrap(np.angle(hs1))
# f_inst_s1 = np.diff(omega_s1)
omega_s2 = np.unwrap(np.angle(fu))
f_inst_s2 = np.diff(fu)
# plt.plot(f_inst_s2, "b")
plt.show()
