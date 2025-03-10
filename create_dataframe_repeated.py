# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:39:32 2023

@author: Simon
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import tkinter as tk
from tkinter import filedialog

def select_directory():
    print('Please select a directory!')
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_root = filedialog.askdirectory()
    return folder_root


def create_dataframe():
    amplitude = np.array([])
    std = np.array([])
    
    # create Dataframe
    # number_of_signals_per_measuring_positions = int(input("Please enter the number of signals per measuring position: "))
    number_of_signals_per_measuring_positions = 25
    # number_of_measuring_positions = int(input("Please enter the number of measuring positions: "))
    number_of_measuring_positions = 35
    # number_of_angles = int(input("Please enter the number of measured angles: "))
    # angles = [float(input(f"Enter angle {i+1} in °: ")) for i in range(0,number_of_angles)]
    angles = (0, 45, 90, 22.5, 67.5)
    # number_of_distances = int(input("Please enter the number of distances per angle: "))
    # distances = [float(input(f"Enter distance {i+1} in m: ")) for i in range(0,number_of_distances)]
    distances = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25)

    cols = ['name', 'angle', 'distance', 'no. of points', 'mean amplitude', 'std.'] + \
            [f'amplitude Burst {i}' for i in range(number_of_signals_per_measuring_positions)]
    df = pd.DataFrame(columns=cols, index=range(number_of_measuring_positions))

    # fill Dataframe
    df.loc[:,"name"] = [f"P{num}" for num in range(1,number_of_measuring_positions+1)]
    df.loc[:,"angle"] = [a for a in angles for i,d in enumerate(distances)]
    df.loc[:,"distance"] = [d for a in angles for i,d in enumerate(distances)]
    df.loc[:,"no. of points"] = [number_of_signals_per_measuring_positions] * number_of_measuring_positions
    
    directory = select_directory()
    os.chdir(directory)
    # current_datetime = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    # os.mkdir('Data_evaluated_' + current_datetime)
    
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
    
    # for path, subdirs, files in tqdm(os.walk(directory)):
    #     for file in files:
    #         if file.endswith("_features.csv"):
    #             os.chdir(path)
    #             data = pd.read_csv(file)
    #             amplitude = np.append(amplitude, data['amplitude'])
    #             std = np.append(std, 0)
                
    #             burst0 = np.append(burst0, data['amplitude'])
    #             burst1 = np.append(burst1, data['amplitude'])
    #             burst2 = np.append(burst2, data['amplitude'])
    #             burst3 = np.append(burst3, data['amplitude'])
    #             burst4 = np.append(burst4, data['amplitude'])
    #             burst5 = np.append(burst5, data['amplitude'])
    #             burst6 = np.append(burst6, data['amplitude'])
    #             burst7 = np.append(burst7, data['amplitude'])
    #             burst8 = np.append(burst8, data['amplitude'])
    #             burst9 = np.append(burst9, data['amplitude'])
    #             burst10 = np.append(burst10, data['amplitude'])
    #             burst11 = np.append(burst11, data['amplitude'])
    #             burst12 = np.append(burst12, data['amplitude'])
    #             burst13 = np.append(burst13, data['amplitude'])
    #             burst14 = np.append(burst14, data['amplitude'])
    #             burst15 = np.append(burst15, data['amplitude'])
    #             burst16 = np.append(burst16, data['amplitude'])
    #             burst17 = np.append(burst17, data['amplitude'])
    #             burst18 = np.append(burst18, data['amplitude'])
    #             burst19 = np.append(burst19, data['amplitude'])
    #             burst20 = np.append(burst20, data['amplitude'])
    #             burst21 = np.append(burst21, data['amplitude'])
    #             burst22 = np.append(burst22, data['amplitude'])
    #             burst23 = np.append(burst23, data['amplitude'])
    #             burst24 = np.append(burst24, data['amplitude'])

    for path, subdirs, files in tqdm(os.walk(directory)):
        for file in files:
            if file.endswith("_features.csv"):
                os.chdir(path)
                data = pd.read_csv(file)
                amplitude = np.append(amplitude, data['energy'])
                std = np.append(std, 0)
                
                burst0 = np.append(burst0, data['energy'])
                burst1 = np.append(burst1, data['energy'])
                burst2 = np.append(burst2, data['energy'])
                burst3 = np.append(burst3, data['energy'])
                burst4 = np.append(burst4, data['energy'])
                burst5 = np.append(burst5, data['energy'])
                burst6 = np.append(burst6, data['energy'])
                burst7 = np.append(burst7, data['energy'])
                burst8 = np.append(burst8, data['energy'])
                burst9 = np.append(burst9, data['energy'])
                burst10 = np.append(burst10, data['energy'])
                burst11 = np.append(burst11, data['energy'])
                burst12 = np.append(burst12, data['energy'])
                burst13 = np.append(burst13, data['energy'])
                burst14 = np.append(burst14, data['energy'])
                burst15 = np.append(burst15, data['energy'])
                burst16 = np.append(burst16, data['energy'])
                burst17 = np.append(burst17, data['energy'])
                burst18 = np.append(burst18, data['energy'])
                burst19 = np.append(burst19, data['energy'])
                burst20 = np.append(burst20, data['energy'])
                burst21 = np.append(burst21, data['energy'])
                burst22 = np.append(burst22, data['energy'])
                burst23 = np.append(burst23, data['energy'])
                burst24 = np.append(burst24, data['energy'])
                
    df['mean amplitude'] = amplitude
    df['std.'] = std
    
    df['amplitude Burst 0'] = burst0
    df['amplitude Burst 1'] = burst1
    df['amplitude Burst 2'] = burst2
    df['amplitude Burst 3'] = burst3
    df['amplitude Burst 4'] = burst4
    df['amplitude Burst 5'] = burst5
    df['amplitude Burst 6'] = burst6
    df['amplitude Burst 7'] = burst7
    df['amplitude Burst 8'] = burst8
    df['amplitude Burst 9'] = burst9
    df['amplitude Burst 10'] = burst10
    df['amplitude Burst 11'] = burst11
    df['amplitude Burst 12'] = burst12
    df['amplitude Burst 13'] = burst13
    df['amplitude Burst 14'] = burst14
    df['amplitude Burst 15'] = burst15
    df['amplitude Burst 16'] = burst16
    df['amplitude Burst 17'] = burst17
    df['amplitude Burst 18'] = burst18
    df['amplitude Burst 19'] = burst19
    df['amplitude Burst 20'] = burst20
    df['amplitude Burst 21'] = burst21
    df['amplitude Burst 22'] = burst22
    df['amplitude Burst 23'] = burst23
    df['amplitude Burst 24'] = burst24
    
    df.to_csv(r"G:\Acoustic Emission\Daten\evaluated\Platte_in_Material\dataframes\plate_repeated_energy.csv")
    # df.to_csv('dataframe_repeated_energy.csv')

create_dataframe()