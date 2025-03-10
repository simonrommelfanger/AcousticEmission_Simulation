# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:50:50 2023

@author: Simon
"""

import math
from nptdms import TdmsFile
import numpy as np
import os
import pandas as pd
from scipy import stats
from scipy import signal
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

#%%------------------------------------------------------------------------ pol2cart

def pol2cart(radius, angle):
    angle = math.radians(angle)
    x = radius * np.cos(angle)
    x = np.round(x, 2)
    y = radius * np.sin(angle)
    y = np.round(y, 2)
    return(x, y)
  
#%%---------------------------------------------------------------- select_directory

def select_directory():
    print('Please select a directory!')
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_root = filedialog.askdirectory()
    return folder_root
  
    

#%%--------------------------------------------------------------------- data_to_csv

#### Transform Raw-Data in CSV-File ####
def data_to_csv(directory, folder_evaluated, data_acquisition):
    
    os.mkdir(folder_evaluated)
    
#### EXPERIMENTAL MEASUREMENT ####

    if data_acquisition == "1":
        print("\n\
experimental measurement selected")
    

        folder_raw_data = folder_evaluated + str('/raw_data')
        os.mkdir(folder_raw_data)
        for path, subdirs, files in tqdm(os.walk(directory)):
            for file in files:
                if file.endswith(".tdms"):
                    file_only = file
                    file = str(path) + str('/') + str(file)
                    tdms_file = TdmsFile.read(file)
                    data_raw = tdms_file.as_dataframe(time_index=True)
                    tdms_file.close()
                    
                    time = pd.DataFrame({'time': np.array(data_raw.index)})
                    
                    data_raw = data_raw.reset_index(drop=True)
                    
                    name = file_only.split('_')[1]
                    name = str(folder_raw_data) + str('/') + str(
                      name) + "_rawdata.csv"
                    column_names = ["Burst{0}".format(i) for i in 
                                    range(data_raw.shape[1])]
                    
                    data_raw = data_raw.reset_index(drop=True)
                    data_raw.columns = column_names
                    
                    data_raw_new = pd.concat([time, data_raw],
                                             axis = 1, ignore_index=False)
                    
                    data_raw_new.to_csv(name, index=False)

#### 3D-Simulation ####

    elif data_acquisition == '2':
        print("\n\
3D-simulation selected")
        
        #### READ AND FILTER DATA ####
        
        folder_read = folder_evaluated + str('/read')
        os.mkdir(folder_read)
        folder_filtered = folder_evaluated + str('/filtered')
        os.mkdir(folder_filtered)
        
        for path, subdirs, files in tqdm(os.walk(directory)):
            for file in files:
                if file.endswith("_raw.txt"):
                    file_only = file
                    file = str(path) + str('/') + str(file)
                    b = np.array([])
                    
                    with open(file, 'r') as f:
                        content = f.readlines()[8::]
                    
                    data_np = np.zeros((len(content), 2), dtype=float)
                    data = np.zeros((len(content), 2), dtype=float)
                    
                    for j in range(len(content)):
                        a = content[j]
                        b = np.append(b, np.array(np.char.split(a)))
                        data_np[j] = b[j]
                        
                    for k in range(np.shape(data_np)[0]):
                        for l in range(np.shape(data_np)[1]):
                            data[k][l] = float(str(
                              data_np[k][l]).replace(',', '.'))
                    
                    time = data[:,0]
                    signal_data = data[:,1]
                                
                    signal_read = np.c_[time, signal_data]
                    
                    name_read = file_only.split('_')[1]
                    name_read = str(folder_read) +str('/') + str(
                      name_read) + str('_read.txt')
                    np.savetxt(name_read, signal_read)
                    
                    timestep = time[1]-time[0]
                    fs = 1/timestep
                    
                    sos = signal.butter(2, 20000, 'hp',
                                        fs = fs, output='sos')
                    signal_filtered = signal.sosfiltfilt(sos, signal_data)
                    
                    signal_filtered = np.c_[time, signal_filtered]

                    name_filtered = file_only.split('_')[1]
                    name_filtered = str(folder_filtered) +str('/') + str(
                      name_filtered) + str('_filtered.txt')
                    np.savetxt(name_filtered, signal_filtered)
                    
        
        folder_raw_data = folder_evaluated + str('/raw_data')
        os.mkdir(folder_raw_data)
        for path, subdirs, files in tqdm(os.walk(folder_filtered)):
            for file in files:
                if file.endswith("_filtered.txt"):
                    file_only = file
                    file = str(path) + str('/') + str(file)
                    data_raw = np.loadtxt(file)
                    time = data_raw[:,0]
                    signal_raw = data_raw[:,1]
                    data_in_df = pd.DataFrame({
                        "time": time,
                        "signal": signal_raw})
                    
                    name = file_only.split('_')[0]
                    name = str(folder_raw_data) + str('/') + str(
                      name) + "_rawdata.csv"
                    data_in_df.to_csv(name, index=False)

#### Other Methods ####

    else:
        print("\n\
Method not supported")
    
    print("\n\
DATA HAS BEEN TRANSFORMED TO CSV!\n\
")
    return
#%%-------------------------------------------------------------- correct_nan

def correct_nan(folder_raw_data):
    
    files_with_anomalies = np.array([])
    for file in tqdm(os.listdir(folder_raw_data)):
        if file.endswith('_rawdata.csv'):
            file_only = file
            file = str(folder_raw_data) + str('/') + str(file)
            data_raw = pd.read_csv(file)
            
            data_raw_time = data_raw.iloc[:, 0]
            data_raw_bursts = data_raw.iloc[:, 1:]
            
            nan_presence = [row.isnull().values.any() for index,
                            row in data_raw_bursts.iloc[:20,:].iterrows()]
            if True in nan_presence:
                files_with_anomalies = np.append(files_with_anomalies, file)
                nan_row_index = np.where(data_raw_bursts.isnull().iloc[:10,
                                                        :] == True)[0][0]
                len_data_raw = data_raw_bursts.shape[1]
                num_nan_values = len(np.where(data_raw_bursts.iloc[
                  nan_row_index, :].isnull() == True)[0])
                num_non_nan_values = len_data_raw - num_nan_values
                if (num_nan_values/len_data_raw) < 0.2:
                    anomalies = np.where(np.isnan(data_raw_bursts.iloc[
                      nan_row_index,:]) == True)[0]
                    data_raw_bursts = data_raw_bursts.drop(
                      data_raw_bursts.columns[anomalies], axis=1)
                    data_raw_bursts = data_raw_bursts.dropna(
                      axis=0, how="all")
                elif (num_nan_values/len_data_raw) > 0.8:
                    anomalies = np.where(np.isnan(
                      data_raw_bursts.iloc[nan_row_index,:]) == False)[0]
                    data_raw_bursts = data_raw_bursts.drop(
                      data_raw_bursts.columns[anomalies], axis=1)
                    data_raw_bursts = data_raw_bursts.dropna(
                      axis=0, how="all")
            
            diff = np.mean(np.diff(data_raw_time))
            time = np.linspace(0, (data_raw_bursts.shape[0]-1)
                               *diff, data_raw_bursts.shape[0])
            data_time = pd.DataFrame({'time': time})
            
            name = file_only.split('_')[0]
            name = str(folder_raw_data) + str('/') + str(
              name) + "_rawdata.csv"
            
            data_raw_bursts.to_csv(name, index=False)
            
            data_raw_bursts = pd.read_csv(name)
            data_raw_new = pd.concat([data_time, data_raw_bursts], axis = 1)
            data_raw_new.to_csv(name, index=False)
            
    print('Files containing nan-rows will be edited so the \
          nan-rows will be deleted. Following files containes nan-rows:\n')
    for element in files_with_anomalies:
        print(element)

#%%---------------------------------------------------------- raw_data_filter

#### Raw-Data Filter ####

def raw_data_filter(folder_raw_data, sampling_rate):
    filter_method = input("Select Filter Method (default: Basic):\n\
(1) Basic\n\
(2) Length\n\
(3) Number\n\
")

    if filter_method == '1':
        print("\n\
Basic Filter selected\n\
")    
        pre_trigger = float(input(
          "Please inpute pre-trigger length (default: 400)):\n\
"))
        dt = sampling_rate


    elif filter_method == '2':
        print("\n\
Length Filter selected\n\
")

    elif filter_method == '3':
        print("\n\
Number Filter selected\n\
")

    else:
        print("\n\
No supported method selected!\n\
")

    folder_signals = folder_evaluated + str('/signals')
    os.mkdir(folder_signals)

    for file in tqdm(os.listdir(folder_raw_data)):
        if file.endswith('_rawdata.csv'):
            file_only = file
            file = str(folder_raw_data) + str('/') + str(file)
            data_raw = pd.read_csv(file)
            
            data_raw_time = data_raw.iloc[:, 0]
            data_raw_bursts = data_raw.iloc[:, 1:]
        
            if filter_method == '1':
                drop_list = np.array([])

                df_properties = pd.DataFrame(columns=["length",
                                "max. amplitude", "total energy"],
                                index=range(data_raw_bursts.shape[1]))

            # Filter data
                for i in range(data_raw_bursts.shape[1]):
                    array = data_raw_bursts[data_raw_bursts.columns[i]]
                    array = array[~np.isnan(array)]
                    df_properties["length"].iloc[i] = array.shape[0]
                    df_properties["max. amplitude"].iloc[i] = max(
                      array) + np.abs(min(array))
                    df_properties["total energy"].iloc[i] = np.sum(
                      np.power(array, 2) * dt)

            # Conditions
                for i in range(df_properties.shape[0]):
                    if (df_properties["max. amplitude"].iloc[i] < np.max(
                        df_properties["max. amplitude"]) * 0.4) or \
                        (df_properties["length"].iloc[i] < round(
                          data_raw_bursts.shape[0] * 0.4)) or \
                            (df_properties["total energy"].iloc[i] < np.max(
                              df_properties["total energy"]) * 0.4):
                                drop_list = np.append(drop_list, i)

                data_filtered = data_raw_bursts.drop(labels=[
                  data_raw_bursts.columns[drop_list[i]] for i in range(len(
                    drop_list))], axis="columns")

            
        
            elif filter_method == '2':
                print("Length Filter not yet implemented")
            
            elif filter_method == '3':
                print("Number Filter not yet implemented")
        
            name = file_only.split('_')[0]
            name = str(folder_signals) + str('/') + str(
              name) + '_signals.csv'
            
            data_filtered_new = pd.concat([data_raw_time,
                            data_filtered], axis = 1, ignore_index=False)
            
            data_filtered_new.to_csv(name, index=False)

    print("\n\
NON-SIGNAL DATA HAS BEEN REMOVED!\n\
")
    return
#%%---------------------------------------------------------------- summarize

#### Summarize Data ####

def summarize(folder_signals):
    folder_summarized = folder_evaluated + str('/summarized')
    os.mkdir(folder_summarized)

    for files in tqdm(os.walk(folder_signals)):
        names = files[2]
        for i in range(len(names)):
            names[i] = names[i].split('_')[0]
        for i in range(len(names)):
            position = names[i][:-1]
            indices = np.array([])
            for j in range(len(names)):
                if names[j][:-1] == position:
                    indices = np.append(indices, j)
            if len(indices) == 2:
                datei_1 = str(folder_signals) + str('/') + names[
                  int(indices[0])] + ('_signals.csv')
                datei_2 = str(folder_signals) + str('/') + names[
                  int(indices[1])] + ('_signals.csv')
                data_1 = pd.read_csv(datei_1)
                data_2 = pd.read_csv(datei_2)
                
                data_1_time = data_1.iloc[:, 0]
                data_1_bursts = data_1.iloc[:, 1:]
                
                data_2_time = data_2.iloc[:, 0]
                data_2_bursts = data_2.iloc[:, 1:]
                
                if len(data_1_time) < len(data_2_time):
                    data_time = data_2_time
                else:
                    data_time = data_1_time
                
                number_of_bursts_1 = data_1_bursts.shape[1]
                number_of_bursts_2 = data_2_bursts.shape[1]
                number_of_bursts = number_of_bursts_1 + number_of_bursts_2
                columns_1 = [f'Burst{i}' for i in range(number_of_bursts_1)]
                columns_2 = [f'Burst{i}' for i in range(
                  number_of_bursts_1, number_of_bursts)]
                
                data_1_new = pd.DataFrame(columns=columns_1)
                data_2_new = pd.DataFrame(columns=columns_2)
                
                for i in range(number_of_bursts_1):
                    data_1_new.iloc[:,i] = data_1_bursts.iloc[:,i] 
                    
                for i in range(number_of_bursts_2):
                    data_2_new.iloc[:,i] = data_2_bursts.iloc[:,i]

                data_new_bursts = pd.concat([data_1_new, data_2_new], axis=1)
                data_new = pd.concat([data_time, data_new_bursts], axis=1)
                file_new = str(folder_summarized) + str('/') + names[
                  int(indices[0])][:-1] + ('_signals.csv')
                data_new.to_csv(file_new, index=False)
            
            elif len(indices) == 3:
                datei_1 = str(folder_signals) + str('/') + names[
                  int(indices[0])] + ('_signals.csv')
                datei_2 = str(folder_signals) + str('/') + names[
                  int(indices[1])] + ('_signals.csv')
                datei_3 = str(folder_signals) + str('/') + names[
                  int(indices[2])] + ('_signals.csv')
                data_1 = pd.read_csv(datei_1)
                data_2 = pd.read_csv(datei_2)
                data_3 = pd.read_csv(datei_3)
                
                data_1_time = data_1.iloc[:, 0]
                data_1_bursts = data_1.iloc[:, 1:]
                
                data_2_time = data_2.iloc[:, 0]
                data_2_bursts = data_2.iloc[:, 1:]
                
                data_3_time = data_3.iloc[:, 0]
                data_3_bursts = data_3.iloc[:, 1:]
                
                if len(data_1_time) < len(data_2_time):
                    data_time_1_2 = data_2_time
                else:
                    data_time_1_2 = data_1_time
                    
                if len(data_time_1_2) < len(data_3_time):
                    data_time = data_3_time
                else:
                    data_time = data_time_1_2
            
                number_of_bursts_1 = data_1_bursts.shape[1]
                number_of_bursts_2 = data_2_bursts.shape[1]
                number_of_bursts_3 = data_3_bursts.shape[1]
                number_of_bursts = number_of_bursts_1 + number_of_bursts_2
                number_of_bursts = number_of_bursts  + number_of_bursts_3
                columns_1 = [f'Burst{i}' for i in range(number_of_bursts_1)]
                columns_2 = [f'Burst{i}' for i in range(number_of_bursts_1,
                  number_of_bursts_1 + number_of_bursts_2)]
                columns_3 = [f'Burst{i}' for i in range(
                  number_of_bursts_1 + number_of_bursts_2, number_of_bursts)]
            
                data_1_new = pd.DataFrame(columns=columns_1)
                data_2_new = pd.DataFrame(columns=columns_2)
                data_3_new = pd.DataFrame(columns=columns_3)

                for i in range(number_of_bursts_1):
                    data_1_new.iloc[:,i] = data_1_bursts.iloc[:,i] 
                
                for i in range(number_of_bursts_2):
                    data_2_new.iloc[:,i] = data_2_bursts.iloc[:,i]
                
                for i in range(number_of_bursts_3):
                    data_3_new.iloc[:,i] = data_2_bursts.iloc[:,i]

                data_new_bursts = pd.concat([
                  data_1_new, data_2_new, data_3_new], axis=1)
                data_new = pd.concat([data_time, data_new_bursts], axis=1)
                file_new = str(folder_summarized) + str('/') + names[
                  int(indices[0])][:-1] + ('_signals.csv')
                data_new.to_csv(file_new, index=False)
            
            elif len(indices) == 4:
                datei_1 = str(folder_signals) + str('/') + names[
                  int(indices[0])] + ('_signals.csv')
                datei_2 = str(folder_signals) + str('/') + names[
                  int(indices[1])] + ('_signals.csv')
                datei_3 = str(folder_signals) + str('/') + names[
                  int(indices[2])] + ('_signals.csv')
                datei_4 = str(folder_signals) + str('/') + names[
                  int(indices[3])] + ('_signals.csv')
                data_1 = pd.read_csv(datei_1)
                data_2 = pd.read_csv(datei_2)
                data_3 = pd.read_csv(datei_3)
                data_4 = pd.read_csv(datei_4)
                
                data_1_time = data_1.iloc[:, 0]
                data_1_bursts = data_1.iloc[:, 1:]
                
                data_2_time = data_2.iloc[:, 0]
                data_2_bursts = data_2.iloc[:, 1:]
                
                data_3_time = data_3.iloc[:, 0]
                data_3_bursts = data_3.iloc[:, 1:]
                
                data_4_time = data_4.iloc[:, 0]
                data_4_bursts = data_4.iloc[:, 1:]
                
                if len(data_1_time) < len(data_2_time):
                    data_time_1_2 = data_2_time
                else:
                    data_time_1_2 = data_1_time
                
                if len(data_3_time) < len(data_4_time):
                    data_time_3_4 = data_4_time
                else:
                    data_time_3_4 = data_3_time
                    
                if len(data_time_1_2) < len(data_time_3_4):
                    data_time = data_time_3_4
                else:
                    data_time = data_time_1_2
                
                number_of_bursts_1 = data_1_bursts.shape[1]
                number_of_bursts_2 = data_2_bursts.shape[1]
                number_of_bursts_3 = data_3_bursts.shape[1]
                number_of_bursts_4 = data_4_bursts.shape[1]
                number_of_bursts = number_of_bursts  + number_of_bursts_3
                number_of_bursts = number_of_bursts  + number_of_bursts_4
                columns_1 = [f'Burst{i}' for i in range(number_of_bursts_1)]
                columns_2 = [f'Burst{i}' for i in range(number_of_bursts_1, 
                  number_of_bursts_1 + number_of_bursts_2)]
                columns_3 = [f'Burst{i}' for i in range(
                  number_of_bursts_1 + number_of_bursts_2,
                  number_of_bursts_1 + number_of_bursts_2 + 
                  number_of_bursts_3)]
                columns_4 = [f'Burst{i}' for i in range(
                  number_of_bursts_1 + number_of_bursts_2 + 
                  number_of_bursts_3, number_of_bursts)]
            
                data_1_new = pd.DataFrame(columns=columns_1)
                data_2_new = pd.DataFrame(columns=columns_2)
                data_3_new = pd.DataFrame(columns=columns_3)
                data_4_new = pd.DataFrame(columns=columns_4)

                for i in range(number_of_bursts_1):
                    data_1_new.iloc[:,i] = data_1_bursts.iloc[:,i] 
                
                for i in range(number_of_bursts_2):
                    data_2_new.iloc[:,i] = data_2_bursts.iloc[:,i]
                
                for i in range(number_of_bursts_3):
                    data_3_new.iloc[:,i] = data_3_bursts.iloc[:,i]
                
                for i in range(number_of_bursts_4):
                    data_4_new.iloc[:,i] = data_4_bursts.iloc[:,i]
                
                data_new_bursts = pd.concat([
                  data_1_new, data_2_new, data_3_new, data_4_new], axis=1)
                data_new = pd.concat([data_time, data_new_bursts], axis=1)
                file_new = str(folder_summarized) + str('/') + names[
                  int(indices[0])][:-1] + ('_signals.csv')
                data_new.to_csv(file_new, index=False)
            

    print("\n\
DATA BELONGING TO THE SAME MEASURING POINT HAS BEEN MERGED!\n\
")
    return
            
#%%------------------------------------------------------------ get_features

def get_features(folder_summarized, sampling_rate, features):
    
    folder_features = folder_evaluated + str('/features')
    os.mkdir(folder_features)
    
    columns = ['angle', 'radius', 'x', 'y'] + features
    
    # Partial Power 0:   0 -  50 kHz
    # Partial Power 1:  50 - 150 kHz
    # Partial Power 2: 150 - 250 kHz
    # Partial Power 3: 250 - 350 kHz
    # Partial Power 4: 350 - 450 kHz
    # Partial Power 5: 450 - 500 kHz
    
    for file in tqdm(os.listdir(folder_summarized)):
        if file.endswith('_signals.csv'):
            file_only = file
            file = str(folder_summarized) + str('/') + str(file)
            data_signals = pd.read_csv(file)
            
            data_signals_time = data_signals.iloc[:, 0]
            data_signals_bursts = data_signals.iloc[:, 1:]
            
            timestep = np.mean(np.diff(data_signals_time))
            
            data_features = pd.DataFrame(columns = columns,
                                         index = ['Burst{0}'.format(i)
                              for i in range(data_signals_bursts.shape[1])])
            
            for i in range(data_signals_bursts.shape[1]):
                signal = data_signals_bursts.iloc[:, i]
                signal = signal[~np.isnan(signal)]
                
                # features time domain                
                amplitude = np.max(np.absolute(signal))
                energy = np.sum(np.power(signal, 2) * timestep)
                rise_time = np.argmax(np.absolute(signal)) * timestep
                rise_angle = np.arctan(amplitude / rise_time)
                
                # features frequency domain
                N = len(signal)
                        
                signal_fft = np.fft.fft(signal)
                signal_fft = signal_fft[:int(N/2)]
                frequency = np.fft.fftfreq(N,d = timestep)
                frequency = frequency[:int(N/2)]
                
                magnitude = np.abs(signal_fft)
                magnitude_normed = 2*magnitude/N
                
                max_magnitude_index = np.where(
                  magnitude_normed == np.max(magnitude_normed[1:]))
                peak_frequency = frequency[max_magnitude_index][0]

                power = np.abs(signal_fft)**2
                power_normed = 2*power/N
                
                psd = 2*timestep/N*(np.abs(signal_fft))**2
                energy_fft = np.sum(psd)
                
                frequency_centroid = np.sum(np.divide(np.multiply(
                  magnitude_normed, frequency), np.sum(magnitude_normed)))
                
                weighted_peak_frequency = np.sqrt(
                  peak_frequency*frequency_centroid)
                
                frequency_start = 0
                frequency_0 = min(min(np.where(50000 <= frequency)))
                frequency_1 = min(min(np.where(150000 <= frequency)))
                frequency_2 = min(min(np.where(250000 <= frequency)))
                frequency_3 = min(min(np.where(350000 <= frequency)))
                frequency_4 = min(min(np.where(450000 <= frequency)))
                frequency_5 = len(frequency)-1
                
                partial_power_0 = np.sum(psd[
                  frequency_start: frequency_0])/energy_fft
                partial_power_1 = np.sum(psd[
                  frequency_0: frequency_1])/energy_fft
                partial_power_2 = np.sum(psd[
                  frequency_1: frequency_2])/energy_fft
                partial_power_3 = np.sum(psd[
                  frequency_2: frequency_3])/energy_fft
                partial_power_4 = np.sum(psd[
                  frequency_3: frequency_4])/energy_fft
                partial_power_5 = np.sum(psd[
                  frequency_4: frequency_5])/energy_fft
                
                angle = np.float32(file_only.split('R')[0][1::])
                radius = np.float32(file_only.split('R')[1].split('_')[0])
                x, y = pol2cart(radius, angle)
                
                data_features["angle"].loc["Burst{0}".format(i)] = angle
                data_features["radius"].loc["Burst{0}".format(i)] = radius
                data_features["x"].loc['Burst{0}'.format(i)] = x
                data_features["y"].loc['Burst{0}'.format(i)] = y
                
                data_features["amplitude"].loc[
                  "Burst{0}".format(i)] = amplitude
                data_features["energy"].loc["Burst{0}".format(i)] = energy
                data_features["rise_time"].loc[
                  "Burst{0}".format(i)] = rise_time
                data_features["rise_angle"].loc[
                  "Burst{0}".format(i)] = rise_angle
                
                data_features["energy_fft"].loc[
                  "Burst{0}".format(i)] = energy_fft
                data_features["frequency_centroid"].loc[
                  "Burst{0}".format(i)] = frequency_centroid
                data_features["peak_frequency"].loc[
                  "Burst{0}".format(i)] = peak_frequency
                data_features["weighted_peak_frequency"].loc[
                  "Burst{0}".format(i)] = weighted_peak_frequency
                data_features["partial_power_0"].loc[
                  "Burst{0}".format(i)] = partial_power_0
                data_features["partial_power_1"].loc[
                  "Burst{0}".format(i)] = partial_power_1
                data_features["partial_power_2"].loc[
                  "Burst{0}".format(i)] = partial_power_2
                data_features["partial_power_3"].loc[
                  "Burst{0}".format(i)] = partial_power_3
                data_features["partial_power_4"].loc[
                  "Burst{0}".format(i)] = partial_power_4
                data_features["partial_power_5"].loc[
                  "Burst{0}".format(i)] = partial_power_5
                
            name = file_only.split('_')[0]
            name = str(folder_features) + str('/') + str(
              name) + '_features.csv'
            data_features.to_csv(name, index=False)
     
    print("\n\
FEATURES HAVE BEEN CALCULATED!\n\
")
    return

#%%------------------------------------------------- features_outlier_removal

def features_outlier_removal(folder_features, features):
    
    q1_toa=0.1
    q3_toa=0.9
    q1_fea=0.25
    q3_fea=0.75
    zlim_toa=5e-1
    zlim_fea=1
    
    method = float(input("Select Filtering Method:\n\
(1) Quantile\n\
(2) Z-Score\n\
"))
    
    folder_features_filtered = folder_evaluated + str('/features_filtered')
    os.mkdir(folder_features_filtered)
    
    for file in tqdm(os.listdir(folder_features)):
        if (file.endswith('_features.csv') or file.endswith('_TOA.csv')):
            file_only = file
            file = str(folder_features) + str('/') + str(file)
            data_features = pd.read_csv(file)
            
            if method == 1:
                if file.endswith("TOA.csv"):
                    Q1 = data_features["TOA"].quantile(q1_toa)
                    Q3 = data_features["TOA"].quantile(q3_toa)
                    IQR = Q3 - Q1
                    outlier = np.where((data_features < (
                        Q1 - 1.5 * IQR)) |(data_features > (
                        Q3 + 1.5 * IQR)) == True)[0]
                    for i in outlier:
                        data_features = data_features.replace(
                          data_features["TOA"].iloc[i], np.nan)
                        
                elif file.endswith("features.csv"):
                    for element in features:
                        Q1 = data_features[element].quantile(q1_fea)
                        Q3 = data_features[element].quantile(q3_fea)
                        IQR = Q3 - Q1
                        outlier = np.where((data_features[element] < (
                          Q1 - 1.5 * IQR)) | (data_features[element] > (
                                             Q3 + 1.5 * IQR)) == True)[0]
                        for i in range(len(features)):
                            if element == features[i]:
                                for j in outlier:
                                    data_features = data_features.replace(
                                      data_features[element].iloc[j], np.nan)
            
            if method == 2:
                if file.endswith("TOA.csv"):
                    zscore = stats.zscore(data_features["TOA"])
                    outlier = np.where(np.abs(zscore) > zlim_toa)[0]
                    for i in outlier:
                        data_features = data_features.replace(
                          data_features["TOA"].iloc[i], np.nan)
                
                elif file.endswith("features.csv"):
                    for element in features:
                        zscore = stats.zscore(data_features[element])
                        outlier = np.where(np.abs(zscore) > zlim_fea)[0]
                        for i in range(len(features)):
                            if element == features[i]:
                                for j in outlier:
                                    data_features = data_features.replace(
                                      data_features[element].iloc[j], np.nan)
    
            name = file_only.split('_')[0]
            name = str(folder_features_filtered) + str('/') + str(
              name) + '_features_filtered.csv'
            data_features.to_csv(name, index=False)
    
    print("\n\
FEATURES HAVE BEEN FILTERED!\n\
")
    return

#%%----------------------------------------------------------- move_files

def move_files():
    
    move = True
    
    print('\n\
Please select the DESTINATION folder!')
    destiny_folder = select_directory()
    
    while move == True:
        print('\n\
Please select the STARTING folder!')
        start_folder = select_directory()
    
        for file in tqdm(os.listdir(start_folder)):
            if file.endswith('_features_filtered.csv'):
                file_only = file
                file = str(start_folder) + str('/') + str(file_only)
                data_copy = pd.read_csv(file)
                
                name_new = str(destiny_folder) + str('/') + str(file_only)
                
                data_copy.to_csv(name_new, index=False)
                
                print("File "+str(file)+" moved to "+str(destiny_folder))
        
        more = input("Would you like to move more files? [y/n]")
        
        if more == "n":
            move = False
    
    return

#%%--------------------------------------------------------- create_dataframe

def create_dataframe(data_acquisition, features):
    
    print("features available:\n\
", features)    
    feature_type = input("Select which feature to extract!\n\
")
    
    default_angles = [0, 22.5, 45, 67.5, 90]
    default_distances = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
    
    number_of_signals_per_measuring_positions = int(input("Please enter \
      the number of signals per measuring position (default: 25): "))
    number_of_angles = int(input(
      "Please enter the number of measured angles (default: 5): "))
    angles = [float(input(f"Enter angle {i+1} in ° (default: "+str(
      default_angles[i])+"): ")) for i in range(0,number_of_angles)]
    number_of_distances = int(input(
      "Please enter the number of distances per angle (default: 7): "))
    distances = [float(input(f"Enter distance {i+1} in m (default: "+str(
      default_distances[i])+"): ")) for i in range(0,number_of_distances)]
    number_of_measuring_positions = number_of_angles * number_of_distances
    
    
    cols = ['name', 'angle', 'distance', 'no. of points', 'mean_'+str(
      feature_type), 'std.'] + \
           [str(feature_type)+f'_Burst{i}' for i in range(
             number_of_signals_per_measuring_positions)]
    data = pd.DataFrame(columns=cols, index=range(
      number_of_measuring_positions))
    
    data.loc[:,"name"] = [f"P{num}" for num in range(
      1,number_of_measuring_positions+1)]
    data.loc[:,"angle"] = [a for a in angles for i,d in enumerate(distances)]
    data.loc[:,"distance"] = [d for a in angles for i,
                              d in enumerate(distances)]
    data.loc[:,"no. of points"] = [number_of_signals_per_measuring_positions
                                   ] * number_of_measuring_positions
    
    folder_dataframes_raw = select_directory()
    
    # folder_dataframes = folder_evaluated + str('/dataframes')
    # os.mkdir(folder_dataframes)
    
#### EXPERIMENTAL MEASUREMENT ####

    if data_acquisition == '1':
        
        number_of_files = 0
        
        
        for path, subdirs, files in tqdm(os.walk(folder_dataframes_raw)):
            for file in files:
                if file.endswith("_features_filtered.csv"):
                    file = folder_dataframes_raw + str('/') + str(file)
                    data_read = pd.read_csv(file)
                    
                    value = data_read[str(feature_type)]
                    value_mean = np.mean(value)
                    value_std = np.std(value)
                    
                    data.loc[:, 'mean_'+str(feature_type)][
                      number_of_files] = value_mean
                    data.loc[:, 'std.'][number_of_files] = value_std
                    
                    if len(
                      value) < number_of_signals_per_measuring_positions:
                        limit = len(value)
                    else:
                        limit = number_of_signals_per_measuring_positions

                    for i in range(limit):
                        name = str(feature_type)+f'_Burst{i}'
                        data.loc[:, name][number_of_files] = value[i]
                    
                    number_of_files = number_of_files + 1
                    
          
        

#### 3D-Simulation ####
    
    elif data_acquisition == '2':
        
        value_mean = np.array([])
        std = np.array([])
        
        burst0 = np.array([])
        burst1 = np.array([])
        burst2 = np.array([])
        burst3 = np.array([])
        burst4 = np.array([])
        burst5 = np.array([])
        burst6 = np.array([])
        burst7 = np.array([])
        burst8 = np.array([])
        burst9 = np.array([])
        burst10 = np.array([])
        burst11 = np.array([])
        burst12 = np.array([])
        burst13 = np.array([])
        burst14 = np.array([])
        burst15 = np.array([])
        burst16 = np.array([])
        burst17 = np.array([])
        burst18 = np.array([])
        burst19 = np.array([])
        burst20 = np.array([])
        burst21 = np.array([])
        burst22 = np.array([])
        burst23 = np.array([])
        burst24 = np.array([])
        
        for path, subdirs, files in tqdm(os.walk(folder_dataframes_raw)):
            for file in files:
                if file.endswith("_features_filtered.csv"):
                    file = folder_dataframes_raw + str('/') + str(file)
                    data_read = pd.read_csv(file)
                    
                    value = data_read[str(feature_type)]
                    value_mean = np.append(value_mean, value)
                    std = np.append(std, 0)
                    
                    burst0 = np.append(burst0, value)
                    burst1 = np.append(burst1, value)
                    burst2 = np.append(burst2, value)
                    burst3 = np.append(burst3, value)
                    burst4 = np.append(burst4, value)
                    burst5 = np.append(burst5, value)
                    burst6 = np.append(burst6, value)
                    burst7 = np.append(burst7, value)
                    burst8 = np.append(burst8, value)
                    burst9 = np.append(burst9, value)
                    burst10 = np.append(burst10, value)
                    burst11 = np.append(burst11, value)
                    burst12 = np.append(burst12, value)
                    burst13 = np.append(burst13, value)
                    burst14 = np.append(burst14, value)
                    burst15 = np.append(burst15, value)
                    burst16 = np.append(burst16, value)
                    burst17 = np.append(burst17, value)
                    burst18 = np.append(burst18, value)
                    burst19 = np.append(burst19, value)
                    burst20 = np.append(burst20, value)
                    burst21 = np.append(burst21, value)
                    burst22 = np.append(burst22, value)
                    burst23 = np.append(burst23, value)
                    burst24 = np.append(burst24, value)
                    
        data['mean_'+str(feature_type)] = value_mean
        data['std.'] = std
        
        data[str(feature_type)+'_Burst0'] = burst0
        data[str(feature_type)+'_Burst1'] = burst1
        data[str(feature_type)+'_Burst2'] = burst2
        data[str(feature_type)+'_Burst3'] = burst3
        data[str(feature_type)+'_Burst4'] = burst4
        data[str(feature_type)+'_Burst5'] = burst5
        data[str(feature_type)+'_Burst6'] = burst6
        data[str(feature_type)+'_Burst7'] = burst7
        data[str(feature_type)+'_Burst8'] = burst8
        data[str(feature_type)+'_Burst9'] = burst9
        data[str(feature_type)+'_Burst10'] = burst10
        data[str(feature_type)+'_Burst11'] = burst11
        data[str(feature_type)+'_Burst12'] = burst12
        data[str(feature_type)+'_Burst13'] = burst13
        data[str(feature_type)+'_Burst14'] = burst14
        data[str(feature_type)+'_Burst15'] = burst15
        data[str(feature_type)+'_Burst16'] = burst16
        data[str(feature_type)+'_Burst17'] = burst17
        data[str(feature_type)+'_Burst18'] = burst18
        data[str(feature_type)+'_Burst19'] = burst19
        data[str(feature_type)+'_Burst20'] = burst20
        data[str(feature_type)+'_Burst21'] = burst21
        data[str(feature_type)+'_Burst22'] = burst22
        data[str(feature_type)+'_Burst23'] = burst23
        data[str(feature_type)+'_Burst24'] = burst24
        
#### SAVE DATA ####
    folder = folder_dataframes_raw.split('/')
    folder = folder[-1]
    folder_base = folder_dataframes_raw.replace(folder, '')
    
    folder_dataframes = folder_base + str('dataframes')
    name = folder_dataframes + str('/dataframe_') + str(
      feature_type) + str('.csv')

    data.to_csv(name, index=True)
    
    return

#%%--------------------------------------------------------------- remove_nan

def remove_nan():
    
    folder_remove = select_directory()
    
    for file in tqdm(os.listdir(folder_remove)):
        if file.endswith('.csv'):
            file = str(folder_remove) + str('/') + str(file)
            data = pd.read_csv(file)
            for i in range(35):
                feature_single = np.array([])
                
                for j in range(25):
                    feature_single = np.append(
                      feature_single, data.iloc[i][-(j+1)])
                    feature_single = feature_single[
                      ~np.isnan(feature_single)]
                
                if len(feature_single) == 25:
                    pass
                else:
                    dif = 25-len(feature_single)
                    for k in range(dif):
                        randoms = np.random.randint(
                          len(feature_single), size=dif)
                        random_value = np.array([])
                        for l in range(len(randoms)):
                            random_value = np.append(
                              random_value, feature_single[randoms[l]])
                            feature_single_new = feature_single
                            feature_single_new = np.append(
                              feature_single_new, random_value)
                        data.iloc[i, -25:] = feature_single_new
            
            data.to_csv(file, index=False)
            
    return

#%%--------------------------------------------------------------------------

features = ['amplitude', 'energy', 'rise_time', 'rise_angle',
            'energy_fft', 'frequency_centroid', 'peak_frequency',
            'weighted_peak_frequency', 'partial_power_0', 'partial_power_1',
            'partial_power_2', 'partial_power_3', 'partial_power_4',
            'partial_power_5']



sampling_rate = 1e-6

data_acquisition = input(
  "Which methods for data acquisition has been used?\n\
(1) - experimental measurement\n\
(2) - 3D-Simulation\n\
")

directory = select_directory()

folder_evaluated = directory + str('/evaluated')
data_to_csv(directory, folder_evaluated, data_acquisition)
folder_raw_data = folder_evaluated +str('/raw_data')
folder_signals = folder_evaluated + str('/signals')

if data_acquisition == '1':
    correct_nan(folder_raw_data)
    folder_summarized = folder_evaluated + str('/summarized')
elif data_acquisition == '2':
    folder_summarized = folder_signals

raw_data_filter(folder_raw_data, sampling_rate)
folder_features = folder_evaluated + str('/features')

summarize(folder_signals)
get_features(folder_summarized, sampling_rate, features)
features_outlier_removal(folder_features, features)

move_files()
create_dataframe(data_acquisition, features)
remove_nan()
