#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d

np.random.seed(42)

N = 8192
# N = 50

x = np.linspace(-1, 1, N).reshape(-1, 1)
# x = np.random.uniform(-1, 1, (N, 1))

x.sort(axis=0)

y = np.cos(np.pi * abs(x)**1.17 * 15)
# y = np.random.normal(0, 1, (N, 1))
data = x

num_interp_points = 8192
x_reg = np.linspace(x.min(), x.max(), num_interp_points)

f_interp = interp1d(x.squeeze(), y.squeeze(), kind='linear', bounds_error=False, fill_value=0.0)
y_reg = f_interp(x_reg)

fft_vals = np.fft.fft(y_reg)
fft_mag = np.abs(fft_vals)
sample_spacing = x_reg[1] - x_reg[0]
frequencies = np.fft.fftfreq(num_interp_points, d=sample_spacing)

fft_mag_shifted = np.fft.fftshift(fft_mag)
frequencies_shifted = np.fft.fftshift(frequencies)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'o-', label='Original Data')
plt.plot(x_reg, y_reg, 'r--', label='Interpolated Data', alpha=0.7)
plt.title('Original and Interpolated Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(frequencies_shifted, fft_mag_shifted)
plt.title('Fourier Transform (Magnitude)')
plt.xlabel('Frequency (cycles per unit x)')
plt.ylabel('Magnitude')
plt.xlim(0, 200)
# plt.ylim(0, 50)
plt.grid(True)

plt.tight_layout()
plt.show()