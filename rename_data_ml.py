# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:10:25 2024

@author: Simon
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import signal
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

#%%---------------------------------------------------------------------------- select_directory

def select_directory():
    print('Please select a directory!')
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_root = filedialog.askdirectory()
    return folder_root

#%%---------------------------------------------------------------------------- pol2cart

def pol2cart(radius, angle):
    angle = math.radians(angle)
    x = radius * np.cos(angle)
    x = np.round(x, 2)
    y = radius * np.sin(angle)
    y = np.round(y, 2)
    return(x, y)
  
#%%---------------------------------------------------------------------------- rename columns and concat signals

# distances = np.array([8.0,13.0,18.0,23.0,28.0,33.0,38.0,43.0,48.0,53.0,58.0,63.0,68.0,73.0,78.0,83.0,88.0,93.0,98.0,103.0,108.0,113.0,118.0,123.0,128.0,133.0,138.0,
#               143.0,148.0,153.0,158.0,163.0,168.0,173.0,178.0,183.0,188.0,193.0,198.0,203.0,208.0,213.0,218.0,223.0,228.0,233.0,238.0,243.0,248.0,253.0])
# distances = distances - 3
# distances = np.int16(distances)
distances = np.array([25, 50, 75, 100, 150, 200, 250])

directory = select_directory()

folder_save_split = directory.split('/')
# folder_save = folder_save_split[0]+str('/')+folder_save_split[1]+str('/')+folder_save_split[2]+str('/')
folder_save = folder_save_split[0]+str('/')+folder_save_split[1]+str('/')+folder_save_split[2]+str('/')+folder_save_split[3]+str('/')


signals = pd.DataFrame()

for path, subdirs, files in tqdm(os.walk(directory)):
    for file in files:
      if file.endswith(".csv"):
        file_only = file
        file = str(directory) + str('/') + str(file)
        data = pd.read_csv(file)
        
        time = data.loc[:, '0']
        signals_one_file = data.drop(['0'], axis=1)
        
        angle = file_only.split('_')[-1].split('.csv')[0]
        
        j = 0
        column_name_new = []
        for i in signals_one_file.columns:
          new_name = str('A') + angle + str('_R') + str(distances[j])
          column_name_new.append(new_name)
          j = j+1
        signals_one_file = signals_one_file.set_axis(column_name_new, axis=1)
        
        signals = pd.concat([signals, signals_one_file], axis=1)

signals = pd.concat([time, signals], axis=1)
signals = signals.rename(columns={'0': 'time'})

name = folder_save + str('/Data_ML_all_signals.csv')
signals.to_csv(name, index=False)

#%%---------------------------------------------------------------------------- filter

time = signals.loc[:, 'time']
signals_without_time = signals.drop(['time'], axis=1)

timestep = np.mean(np.diff(time))
fs = 1/timestep

sos = signal.butter(2, 20000, 'hp', fs = fs, output='sos')

signals_filtered = pd.DataFrame()
for i in signals_without_time.columns:
  signal_single = signals_without_time.loc[:, i]
  signal_single_filtered = signal.sosfiltfilt(sos, signal_single)
  
  signals_filtered[i] = signal_single_filtered.tolist()

sos_1mhz = signal.butter(2, 1000000, 'lp', fs = fs, output='sos')
signals_filtered_1mhz = pd.DataFrame()
for i in signals_filtered.columns:
  signal_single_1mhz = signals_filtered.loc[:, i]
  signal_single_filtered_1mhz = signal.sosfiltfilt(sos_1mhz, signal_single_1mhz)
  
  signals_filtered_1mhz[i] = signal_single_filtered_1mhz.tolist()

signals_filtered = pd.concat([time, signals_filtered], axis=1)
signals_filtered_1mhz = pd.concat([time, signals_filtered_1mhz], axis=1)

name_filtered = folder_save + str('/Data_ML_all_signals_filtered.csv')
signals_filtered.to_csv(name_filtered, index=False)

name_filtered_1mhz = folder_save + str('/Data_ML_all_signals_filtered_1mhz.csv')
signals_filtered_1mhz.to_csv(name_filtered_1mhz, index=False)

#%%---------------------------------------------------------------------------- features

features = ['amplitude', 'energy', 'rise_time', 'rise_angle', 'energy_fft',
            'frequency_centroid', 'peak_frequency', 'weighted_peak_frequency',
            'partial_power_0', 'partial_power_1', 'partial_power_2',
            'partial_power_3', 'partial_power_4', 'partial_power_5']

columns = ['angle', 'radius', 'x', 'y'] + features

    # Partial Power 0:   0 -  50 kHz
    # Partial Power 1:  50 - 150 kHz
    # Partial Power 2: 150 - 250 kHz
    # Partial Power 3: 250 - 350 kHz
    # Partial Power 4: 350 - 450 kHz
    # Partial Power 5: 450 - 500 kHz
    
time = signals_filtered.loc[:, 'time']
signals_filtered_without_time = signals_filtered.drop(['time'], axis=1)

signals_filtered_features = pd.DataFrame(columns = columns,
                                         index = ['signal_{0}'.format(i) for i in range(signals_filtered_without_time.shape[1])])
j = 0
for i in signals_filtered_without_time:
  signal_filtered_single = signals_filtered_without_time.loc[:, i]
  
  # features time domain                  
  amplitude = np.max(np.absolute(signal_filtered_single))
  energy = np.sum(np.power(signal_filtered_single, 2) * timestep)
  rise_time = np.argmax(np.absolute(signal_filtered_single)) * timestep
  rise_angle = np.arctan(amplitude / rise_time)
  
  # features frequency domain
  N = len(signal_filtered_single)
          
  signal_fft = np.fft.fft(signal_filtered_single)
  signal_fft = signal_fft[:int(N/2)]
  frequency = np.fft.fftfreq(N,d = timestep)
  frequency = frequency[:int(N/2)]
  
  magnitude = np.abs(signal_fft)
  magnitude_normed = 2*magnitude/N
  
  max_magnitude_index = np.where(magnitude_normed == np.max(magnitude_normed[1:]))
  peak_frequency = frequency[max_magnitude_index][0]
  
  power = np.abs(signal_fft)**2
  power_normed = 2*power/N
  
  psd = 2*timestep/N*(np.abs(signal_fft))**2
  energy_fft = np.sum(psd)
  
  frequency_centroid = np.sum(np.divide(np.multiply(magnitude_normed, frequency), np.sum(magnitude_normed)))
  
  weighted_peak_frequency = np.sqrt(peak_frequency*frequency_centroid)
  
  frequency_start = 0
  frequency_0 = min(min(np.where(50000 <= frequency)))
  frequency_1 = min(min(np.where(150000 <= frequency)))
  frequency_2 = min(min(np.where(250000 <= frequency)))
  frequency_3 = min(min(np.where(350000 <= frequency)))
  frequency_4 = min(min(np.where(450000 <= frequency)))
  frequency_5 = len(frequency)-1
  
  partial_power_0 = np.sum(psd[frequency_start: frequency_0])/energy_fft
  partial_power_1 = np.sum(psd[frequency_0: frequency_1])/energy_fft
  partial_power_2 = np.sum(psd[frequency_1: frequency_2])/energy_fft
  partial_power_3 = np.sum(psd[frequency_2: frequency_3])/energy_fft
  partial_power_4 = np.sum(psd[frequency_3: frequency_4])/energy_fft
  partial_power_5 = np.sum(psd[frequency_4: frequency_5])/energy_fft
  
  angle = np.float32(i.split('_')[0].split('A')[-1])
  radius = np.int16(i.split('_')[-1].split('R')[-1])
  x, y = pol2cart(radius, angle)
  
  signals_filtered_features["angle"].loc['signal_{0}'.format(j)] = angle
  signals_filtered_features["radius"].loc['signal_{0}'.format(j)] = radius
  signals_filtered_features["x"].loc['signal_{0}'.format(j)] = x
  signals_filtered_features["y"].loc['signal_{0}'.format(j)] = y
  
  signals_filtered_features["amplitude"].loc['signal_{0}'.format(j)] = amplitude
  signals_filtered_features["energy"].loc['signal_{0}'.format(j)] = energy
  signals_filtered_features["rise_time"].loc['signal_{0}'.format(j)] = rise_time
  signals_filtered_features["rise_angle"].loc['signal_{0}'.format(j)] = rise_angle
  
  signals_filtered_features["energy_fft"].loc['signal_{0}'.format(j)] = energy_fft
  signals_filtered_features["frequency_centroid"].loc['signal_{0}'.format(j)] = frequency_centroid
  signals_filtered_features["peak_frequency"].loc['signal_{0}'.format(j)] = peak_frequency
  signals_filtered_features["weighted_peak_frequency"].loc['signal_{0}'.format(j)] = weighted_peak_frequency
  signals_filtered_features["partial_power_0"].loc['signal_{0}'.format(j)] = partial_power_0
  signals_filtered_features["partial_power_1"].loc['signal_{0}'.format(j)] = partial_power_1
  signals_filtered_features["partial_power_2"].loc['signal_{0}'.format(j)] = partial_power_2
  signals_filtered_features["partial_power_3"].loc['signal_{0}'.format(j)] = partial_power_3
  signals_filtered_features["partial_power_4"].loc['signal_{0}'.format(j)] = partial_power_4
  signals_filtered_features["partial_power_5"].loc['signal_{0}'.format(j)] = partial_power_5
  
  j = j+1
  
name_filtered_features = folder_save + str('/Data_ML_all_signals_filtered_features.csv')
signals_filtered_features.to_csv(name_filtered_features, index=False)

# j=0  
# for i in signals_filtered.columns:
#   plt.figure(figsize=[13,10])
#   plt.title(i)
#   plt.plot(time, signals_filtered.loc[:, str(i)])
#   plt.show()
#   plt.figure(figsize=[13,10])
#   plt.title(i+str('1 MHz'))
#   plt.plot(time, signals_filtered_1mhz.loc[:, str(i)])
#   plt.show()
#   j=j+1