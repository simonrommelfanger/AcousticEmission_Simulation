# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 17:31:31 2022

@author: Simon
"""

import ntpath
import numpy as np


def read_signal(file, save, folder=''):
    b = np.array([])

    head, tail = ntpath.split(file)
    name = tail[:-4]

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
            data[k][l] = float(str(data_np[k][l]).replace(',', '.'))
    
    time = data[:,0]
    signal = data[:,1]
                
    signal_read = np.c_[time, signal]
    
    if save == True:
        folder = folder + str('/evaluated/')
        np.savetxt(str(folder)+str(name)+'_read.txt', signal_read)
    else:
        None
        
    return name, signal_read
