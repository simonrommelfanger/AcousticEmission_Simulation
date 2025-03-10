# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:07:14 2024

@author: Simon
"""

import os
import pandas as pd
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
  
    
#%%----------------------------------------------------------------------------

directory = select_directory()
data_concated = pd.DataFrame()

for file in tqdm(os.listdir(directory)):
    if file.endswith('_features_filtered.csv'):
      file_read = directory + '/' + file
      data = pd.read_csv(file_read)
      data_concated = pd.concat([data_concated, data], axis=0, ignore_index=True)


folder_save = r"D:\Uni\Train4SHM\Data_ML\Messung"
name_filtered_features = folder_save + str('/Messung_Data_ML_all_signals_filtered_features.csv')
data_concated.to_csv(name_filtered_features, index=False)