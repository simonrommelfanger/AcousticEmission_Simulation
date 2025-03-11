# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:09:22 2022

@author: Simon
"""

import numpy as np
from scipy import signal

def filter(data, save, folder, name):
    
    time = data[:,0]
    signal_raw = data[:,1]
    
    timestep = time[1]-time[0]
    fs = 1/timestep
    
    sos = signal.butter(4, 50000, 'hp', fs = fs, output='sos')
    signal_filtered = signal.sosfilt(sos, signal_raw)
    
    signal_filtered = np.c_[time, signal_filtered]
    
    if save == True:
        folder = folder + str('/evaluated/')
        np.savetxt(str(folder)+str(name)+'_filtered.txt', signal_filtered)
    
    else:
        None
    
    return signal_filtered

#%%----------------------------------------------------------------------------

def filter_forwardbackward(data, save, folder='', name=''):
    
    time = data[:,0]
    signal_raw = data[:,1]
    
    timestep = time[1]-time[0]
    fs = 1/timestep
    
    sos = signal.butter(2, 50000, 'hp', fs = fs, output='sos')
    signal_filtered = signal.sosfiltfilt(sos, signal_raw)
    
    signal_filtered = np.c_[time, signal_filtered]
    
    if save == True:
        folder = folder + str('/evaluated/')
        np.savetxt(str(folder)+str(name)+'_filtered.txt', signal_filtered)
    
    else:
        None
    
    return signal_filtered