# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:39:27 2023

@author: Simon
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

file = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\Data_evaluated_150323-233051\P0.5a_signals.csv"
data = pd.read_csv(file)

signal_raw = np.array(data.iloc[:, 1])
signal_raw = signal_raw[~np.isnan(signal_raw)]

time = np.array(data.iloc[:, 0])[: len(signal_raw)]
dt = time[1]
fs = 1/dt

# sos = signal.butter(2, 1000, 'hp', fs = fs, output='sos')
# signal_filtered = signal.sosfiltfilt(sos, signal_raw)

plt.figure(figsize=[13,10])
plt.plot(time, signal_raw)
# plt.plot(time, signal_filtered)
plt.show()


N = len(signal_raw)

signal_fft = np.fft.fft(signal_raw)
signal_fft = signal_fft[:int(N/2)]
frequency = np.fft.fftfreq(N,d=dt)
frequency = frequency[:int(N/2)]

# signal_filtered_fft = np.fft.fft(signal_filtered)
# signal_filtered_fft = signal_filtered_fft[:int(N/2)]

magnitude = np.abs(signal_fft)
magnitude_normalized = 2*magnitude/N

# magnitude_filtered = np.abs(signal_filtered_fft)

power = (np.abs(signal_fft))**2
power_normalized = 2*power/N

psd = 2*dt/N*(np.abs(signal_fft))**2
# psd_filtered = 2*dt/N*(np.abs(signal_filtered_fft))**2

power_value = np.sum(((np.abs(signal_raw))**2)*dt)
power_value_fft = np.sum(((np.abs(magnitude))**2)*dt)
power_value_psd = np.sum(psd)
# power_value_psd_filtered = np.sum(psd_filtered)

plt.figure(figsize=[13,10])
# plt.plot(time, signal)
plt.plot(frequency, psd)
# plt.plot(frequency, psd_filtered)
plt.show()