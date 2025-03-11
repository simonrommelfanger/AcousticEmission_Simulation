# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:14:35 2022

@author: Simon
"""

import matplotlib.pyplot as plt
# import numpy as np


def plot(name, signal, save, folder):
    
    plt.figure(figsize=[13,10])
    plt.title('Signal '+str(name), size=25, weight='bold')
    plt.plot(signal[:,0], signal[:,1])
    plt.xlabel('Zeit t in s', size=25, weight='bold')
    plt.ylabel('Auslenkung in m', size=25, weight='bold')
    plt.grid('on', linestyle = '--')
    plt.tick_params(labelsize = 25)
    # legend_font = {'family' : 'Arial', 'weight' : 'normal', 'size': 12}
    # plt.legend(prop=legend_font, loc = 'best')
    
    if save == True:
        folder = folder + str('/evaluated/')
        plt.savefig(str(folder)+'/'+str(name)+'.png', bbox_inches='tight', dpi=200)
    
    else:
        None
    
    plt.show()
    
#%%----------------------------------------------------------------------------

def plot_both(name, signal_1, signal_2, save, folder, title='Signal '):
    
    if title == 'Signal ':
        title = 'Signal '+str(name)
    else:
        pass
    
    plt.figure(figsize=[13,10])
    plt.title(title, size=25, weight='bold')
    plt.plot(signal_1[:,0], signal_1[:,1], 'b', label='signal_1')
    plt.plot(signal_2[:,0], signal_2[:,1], 'r', label='signal_2')
    plt.xlabel('Zeit t in s', size=25, weight='bold')
    plt.ylabel('Auslenkung in m', size=25, weight='bold')
    plt.grid('on', linestyle = '--')
    plt.tick_params(labelsize = 25)
    legend_font = {'family' : 'Arial', 'weight' : 'normal', 'size': 12}
    plt.legend(prop=legend_font, loc = 'best')
    
    if save == True:
        folder = folder + str('/evaluated/')
        plt.savefig(str(folder)+str(title)+'_both.png', bbox_inches='tight', dpi=200)
    
    else:
        None
    
    plt.show()

#%%----------------------------------------------------------------------------

def plot_fft(name, signal, save, folder, title='FFT Signal '):
    
    if title == 'FFT Signal ':
        title = 'FFT Signal ' + str(name)
    else:
        pass
    
    time_values = signal[:, 0]
    signal_values = signal[:, 1]
    
    Fs = 1/time_values[1]
    
    spec, freq, _ = plt.magnitude_spectrum(signal_values, Fs, visible=False)
    
    plt.figure(figsize=[13,10])
    plt.title(title, size=25, weight='bold')
    plt.plot(freq/1e3, spec, linewidth=3)
    plt.xlabel('Frequenz [kHz]', size=25, weight='bold')
    plt.ylabel('Magnitude', size=25, weight='bold')
    plt.xlim(0, 2000)
    plt.grid('on', linestyle = '--')
    plt.tick_params(labelsize = 25)
    # legend_font = {'family' : 'Arial', 'weight' : 'normal', 'size': 12}
    # plt.legend(prop=legend_font, loc = 'best')
    
    if save == True:
        folder = folder + str('/evaluated/')
        plt.savefig(str(folder)+str(title)+'_fft.png', bbox_inches='tight', dpi=200)
    else:
        None
    
    plt.show()