# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:09:04 2023

@author: Simon
"""

from filter import filter_forwardbackward
import numpy as np
import os
from read_signal import read_signal
from scipy import signal
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

#%%----------------------------------------------------------------------------

def select_directory():
    print('Please select a directory!')
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    folder_root = filedialog.askdirectory()
    return folder_root

#%%----------------------------------------------------------------------------

print("Select Save-Folder:")
save_folder = select_directory()

save_folder_read = str(save_folder) + '/read'
save_folder_filtered = str(save_folder) + '/filtered'

os.mkdir(save_folder_read)
os.mkdir(save_folder_filtered)

print("Select Folder for evaluation:")
folder = select_directory()

for path, subdirs, files in tqdm(os.walk(folder)):
    for file in files:
        filename = path+str('/')+file
        name, signal_read = read_signal(filename, save=False)
        np.savetxt(str(save_folder_read)+str('/')+str(name)+'_read.txt', signal_read)
        
        signal_filtered = filter_forwardbackward(signal_read, save=False)
        np.savetxt(str(save_folder_filtered)+str('/')+str(name)+'_filtered.txt', signal_filtered)
        
