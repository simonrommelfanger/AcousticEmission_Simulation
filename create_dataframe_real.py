# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:16:01 2023

@author: Simon
"""

import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# %%

def select_directory():
    print('Please select a directory!')
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_root = filedialog.askdirectory()
    return folder_root


# %%

number_of_files = 0

# number_of_signals_per_measuring_positions = int(input("Please enter the number of signals per measuring position: "))
number_of_signals_per_measuring_positions = 25
# number_of_measuring_positions = int(input("Please enter the number of measuring positions: "))
number_of_measuring_positions = 35
# number_of_angles = int(input("Please enter the number of measured angles: "))
# angles = [float(input(f"Enter angle {i+1} in °: ")) for i in range(0,number_of_angles)]
angles = [0, 45, 90, 22.5, 67.5]
# number_of_distances = int(input("Please enter the number of distances per angle: "))
# distances = [float(input(f"Enter distance {i+1} in m: ")) for i in range(0,number_of_distances)]
distances = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]

cols = ['name', 'angle', 'distance', 'no. of points', 'mean amplitude', 'std.'] + \
       [f'amplitude Burst {i}' for i in range(number_of_signals_per_measuring_positions)]
df = pd.DataFrame(columns=cols, index=range(number_of_measuring_positions))

df.loc[:,"name"] = [f"P{num}" for num in range(1,number_of_measuring_positions+1)]
df.loc[:,"angle"] = [a for a in angles for i,d in enumerate(distances)]
df.loc[:,"distance"] = [d for a in angles for i,d in enumerate(distances)]
df.loc[:,"no. of points"] = [number_of_signals_per_measuring_positions] * number_of_measuring_positions

directory = select_directory()
os.chdir(directory)

for path, subdirs, files in tqdm(os.walk(directory)):
    for file in files:
        print(file)
        if file.endswith("_features_filtered.csv"):
            os.chdir(path)
            data = pd.read_csv(file)
            
            amp = data['partial_power_5']
            amp_mean = np.mean(amp)
            amp_std = np.std(amp)
            
            df.loc[:, 'mean amplitude'][number_of_files] = amp_mean
            df.loc[:, 'std.'][number_of_files] = amp_std
            
            if len(amp) < number_of_signals_per_measuring_positions:
                limit = len(amp)
            else:
                limit = number_of_signals_per_measuring_positions

            for i in range(limit):
                name = f'amplitude Burst {i}'
                df.loc[:, name][number_of_files] = amp[i]
            
            number_of_files = number_of_files + 1

df.to_csv(r"G:\Acoustic Emission\Daten\Lennart\Plate\plate_partial_power_5.csv")
