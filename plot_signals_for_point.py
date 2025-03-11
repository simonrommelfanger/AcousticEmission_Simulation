# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 14:30:02 2023

@author: simon
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_all_signals_for_point(file, point, save, folder):
    data = pd.read_csv(file)
    time = np.linspace(0, 1e-6*data.shape[0], data.shape[0])
    
    plt.figure(figsize=[13,10])
    plt.title('signals for measuring point '+str(point), weight='bold')
    plt.xlabel('Time in s', weight='bold')
    plt.ylabel('Amplitude in V', weight='bold') 
    for i in range(data.shape[1]):
        signal = np.array(data.iloc[:, i])
        plt.plot(time, signal)
    
    if save == True:
        plt.savefig(str(folder)+'/signals_P'+str(point)+'.png', bbox_inches='tight', dpi=200)
    
    plt.show()

#%%----------------------------------------------------------------------------

def fft(file, point, save, folder):
    data = pd.read_csv(file)
    dt = 1e-6
    
    plt.figure(figsize=[13,10])
    plt.title('power spectrum densities for measuring point '+str(point), weight='bold')
    plt.xlabel('frequncy in hz', weight='bold')
    plt.ylabel('density', weight='bold')
    
    for i in range(data.shape[1]):
        signal = np.array(data.iloc[:, i])
        signal = signal[~np.isnan(signal)]
        N = len(signal)
        
        signal_fft = np.fft.fft(signal)
        signal_fft = signal_fft[:int(N/2)]
        frequency = np.fft.fftfreq(N,d=dt)
        frequency = frequency[:int(N/2)]
        
        # magnitude = np.abs(signal_fft)
        # magnitude_normalized = 2*magnitude/N
        # power = (np.abs(signal_fft))**2
        # power_normalized = 2*power/N
        
        psd = 2*dt/N*(np.abs(signal_fft))**2
        
        plt.plot(frequency, psd)
    
    if save == True:
        plt.savefig(str(folder)+'/PSD_P'+str(point)+'.png', bbox_inches='tight', dpi=200)
    
    plt.show()

#%%----------------------------------------------------------------------------
#####    0°    #####

file_P0k5 = r"E:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P0.5_signal.csv"
plot_all_signals_for_point(file_P0k5, point='0.5 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# fft(file_P0k5, point='0.5 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# file_P1 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P1_signal.csv"
# plot_all_signals_for_point(file_P1, point='1 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# # fft(file_P1, point='1 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# file_P1k5 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P1.5_signal.csv"
# plot_all_signals_for_point(file_P1k5, point='1.5 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# # fft(file_P1k5, point='1.5 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# file_P2 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P2_signal.csv"
# plot_all_signals_for_point(file_P2, point='2 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# # fft(file_P2, point='2 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# file_P3 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P3_signal.csv"
# plot_all_signals_for_point(file_P3, point='3 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# # fft(file_P3, point='3 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# file_P4 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P4_signal.csv"
# plot_all_signals_for_point(file_P4, point='4 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# # fft(file_P4, point='4 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# file_P5 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals\P5_signal.csv"
# plot_all_signals_for_point(file_P5, point='5 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')
# # fft(file_P5, point='5 - 0°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\0°\signals')

# #%%----------------------------------------------------------------------------
# #####  22.5°   #####

# file_P16 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P16_signal.csv"
# plot_all_signals_for_point(file_P16, point='16 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P16, point='16 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')

# file_P17 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P17_signal.csv"
# plot_all_signals_for_point(file_P17, point='17 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P17, point='17 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')

# file_P18 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P18_signal.csv"
# plot_all_signals_for_point(file_P18, point='18 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P18, point='18 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')

# file_P19 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P19_signal.csv"
# plot_all_signals_for_point(file_P19, point='19 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P19, point='19 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')

# file_P20 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P20_signal.csv"
# plot_all_signals_for_point(file_P20, point='20 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P20, point='20 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')

# file_P21 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P21_signal.csv"
# plot_all_signals_for_point(file_P21, point='21 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P21, point='21 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')

# file_P22 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals\P22_signal.csv"
# plot_all_signals_for_point(file_P22, point='22 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')
# # fft(file_P22, point='22 - 22.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\22.5°\signals')


# #%%----------------------------------------------------------------------------
# #####   45°    #####

# file_P5k5 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P5.5_signal.csv"
# plot_all_signals_for_point(file_P5k5, point='5.5 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P5k5, point='5.5 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# file_P6 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P6_signal.csv"
# plot_all_signals_for_point(file_P6, point='6 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P6, point='6 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# file_P6k5 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P6.5_signal.csv"
# plot_all_signals_for_point(file_P6k5, point='6.5 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P6k5, point='6.5 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# file_P7 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P7_signal.csv"
# plot_all_signals_for_point(file_P7, point='7 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P7, point='7 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# file_P8 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P8_signal.csv"
# plot_all_signals_for_point(file_P8, point='8 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P8, point='8 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# file_P9 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P9_signal.csv"
# plot_all_signals_for_point(file_P9, point='9 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P9, point='9 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# file_P10 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals\P10_signal.csv"
# plot_all_signals_for_point(file_P10, point='10 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')
# # fft(file_P10, point='10 - 45°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\45°\signals')

# #%%----------------------------------------------------------------------------
# #####  67.5°   #####

# file_P23 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P23_signal.csv"
# plot_all_signals_for_point(file_P23, point='23 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P23, point='23 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# file_P24 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P24_signal.csv"
# plot_all_signals_for_point(file_P24, point='24 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P24, point='24 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# file_P25 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P25_signal.csv"
# plot_all_signals_for_point(file_P25, point='25 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P25, point='25 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# file_P26 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P26_signal.csv"
# plot_all_signals_for_point(file_P26, point='26 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P26, point='26 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# file_P27 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P27_signal.csv"
# plot_all_signals_for_point(file_P27, point='27 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P27, point='27 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# file_P28 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P28_signal.csv"
# plot_all_signals_for_point(file_P28, point='28 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P28, point='28 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# file_P29 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals\P29_signal.csv"
# plot_all_signals_for_point(file_P29, point='29 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')
# # fft(file_P29, point='29 - 67.5°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\67.5°\signals')

# #%%----------------------------------------------------------------------------
# #####   90°    #####

# file_P10k5 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P10.5_signal.csv"
# plot_all_signals_for_point(file_P10k5, point='10.5 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P10k5, point='10.5 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')

# file_P11 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P11_signal.csv"
# plot_all_signals_for_point(file_P11, point='11 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P11, point='11 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')

# file_P11k5 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P11.5_signal.csv"
# plot_all_signals_for_point(file_P11k5, point='11.5 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P11k5, point='11.5 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')

# file_P12 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P12_signal.csv"
# plot_all_signals_for_point(file_P12, point='12 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P12, point='12 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')

# file_P13 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P13_signal.csv"
# plot_all_signals_for_point(file_P13, point='13 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P13, point='13 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')

# file_P14 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P14_signal.csv"
# plot_all_signals_for_point(file_P14, point='14 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P14, point='14 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')

# file_P15 = r"G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals\P15_signal.csv"
# plot_all_signals_for_point(file_P15, point='15 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
# # fft(file_P15, point='15 - 90°', save=False, folder=r'G:\Train4SHM\Python_Codes\Andere\RohDaten\Plate\90°\signals')
