# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:36:38 2023

@author: Simon
"""

import datetime
import h5py
import ntpath
import numpy as np
import os
import pandas as pd
from scipy import signal
import tkinter as tk
from tkinter import filedialog
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tqdm import tqdm

#%%----------------------------------------------------------------------------

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

#%%----------------------------------------------------------------------------

def filter(data, save, folder='', name=''):
    
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

#%%----------------------------------------------------------------------------

def features(data):
    zeit = data[:, 0]
    signal = data[:, 1]
    
    dt = zeit[1]
    
    max_amplitude = np.max(np.absolute(signal))
    energy = np.sum(np.power(signal, 2) * dt)
    rise_time = np.argmax(np.absolute(signal)) * dt
    rise_angle = np.arctan(max_amplitude / rise_time)
    
    return max_amplitude, energy, rise_time, rise_angle

#%%----------------------------------------------------------------------------

datentyp = float(input('Wurden die Daten mit einer 2D oder 3D Simulation erstellt?\n\
(1) 2D\n\
(2) 3D\n'))

punkte = int(input('Aus wie vielen Punkten besteht das Signal?\n'))

#%%----------------------------------------------------------------------------

read = True
current_date = datetime.datetime.now().strftime("%d%m%Y")

vorlaufszeit = np.array([])
anstiegszeit = np.array([])
kraft = np.array([])
nachlaufzeit = np.array([])

abstand = np.array([])
winkel = np.array([])
max_amplitude = np.array([])
energy = np.array([])

time = np.zeros((1, punkte))
signal_raw = np.zeros((1, punkte))
signal_filtered = np.zeros((1, punkte))

#%%----------------------------------------------------------------------------

zähl_durchgehend = 0

if datentyp == 1:
    
    while read == True:
        
        # print("Wähle Ordner zum Auswerten:")
        # root = Tk()
        # root.attributes('-topmost',True)
        # root.wm_attributes('-topmost', 1)
        # root.withdraw()
        # folder = askdirectory(title='Select Folder')
        print('Please select a directory!')
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        directory = filedialog.askdirectory()
        
        print(directory)
        
        # vorlaufszeit_einzeln = float(input("Vorlaufszeit:\n"))
        # anstiegszeit_einzeln = float(input("Anstiegszeit:\n"))
        # kraft_einzeln = float(input("maximale Kraft:\n"))
        # nachlaufzeit_einzeln = float(input("Nachlaufzeit:\n"))
        
        print('\n\
Signale werden eingelesen!')
        
        for path, subdirs, files in tqdm(os.walk(directory)):
            folder = path
            folder_evaluated = folder + str('/evaluated')
            os.mkdir(folder_evaluated)
            
            for file in files:
                if file.endswith('mm.txt'):
                
                    filename = path + '/' + file
                    
                    folder_name = folder.split('/')[3]
                    parameters = folder_name.split('_')
                    
                    vorlaufszeit_einzeln = float(parameters[0][28:-2])
                    anstiegszeit_einzeln = float(parameters[1][7:-2])
                    kraft_einzeln = float(parameters[2][5:-1])
                    nachlaufzeit_einzeln = float(parameters[3][8:-2])
                    
                    vorlaufszeit = np.append(vorlaufszeit, vorlaufszeit_einzeln)
                    anstiegszeit = np.append(anstiegszeit, anstiegszeit_einzeln)
                    kraft = np.append(kraft, kraft_einzeln)
                    nachlaufzeit = np.append(nachlaufzeit, nachlaufzeit_einzeln)
                    
                    name, data_read = read_signal(filename, save=True, folder=folder)
                    data_filtered = filter(data_read, save=True, folder=folder, name=name)
                    max_amplitude_einzeln, energy_einzeln, rise_tim_einzelne, rise_angle_einzeln = features(data_filtered)
                    
                    time_einzeln = np.array([data_read[:, 0]])
                    signal_raw_einzeln = np.array([data_read[:, 1]])
                    signal_filtered_einzeln = np.array([data_filtered [:, 1]])
                    abstand_einzeln = float(name[-5:-2])
                    winkel_einzeln = np.nan
                    
                    time = np.vstack((time, time_einzeln))
                    signal_raw = np.vstack((signal_raw, signal_raw_einzeln))
                    signal_filtered = np.vstack((signal_filtered, signal_filtered_einzeln))
                    
                    abstand = np.append(abstand, abstand_einzeln)
                    winkel = np.append(winkel, winkel_einzeln)
                    max_amplitude = np.append(max_amplitude, max_amplitude_einzeln)
                    energy = np.append(energy, energy_einzeln)
                    
                    zähl_durchgehend = zähl_durchgehend +1
                    
                    print(folder, name + str(' done'))
                    print('Bisher wurden '+str(zähl_durchgehend)+' Signale eingelesen')
        
        

        read = False
            
        print("Wähle Ordner zum Abspeichern der hdf5-Datei:")
        root = Tk()
        root.attributes('-topmost',True)
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        folder_save = askdirectory(title='Select Folder')
        print(folder_save)
        
        time = np.delete(time, 0, 0)
        signal_raw = np.delete(signal_raw, 0, 0)
        signal_filtered = np.delete(signal_filtered, 0, 0)
            
        table = pd.DataFrame({"Vorlaufszeit": vorlaufszeit,
                              "Anstiegszeit": anstiegszeit,
                              "Kraft": kraft,
                              "Nachlaufzeit": nachlaufzeit,
                              "Abstand": abstand,
                              "Winkel": winkel,
                              "max_amplitude": max_amplitude,
                              "energy": energy})
        
        save_name = folder_save+str('\data_2D_')+current_date
        
        with h5py.File(save_name+str(".hdf5"), "a") as f:
            f.create_dataset("time", data = time)
            f.create_dataset("signal_raw", data = signal_raw)
            f.create_dataset("signal_filtered", data = signal_filtered)
            f.create_dataset("table", data=table)
        
        print('Es wurden insgesamt '+str(zähl_durchgehend)+' Signale abgespeichert')

            
            
        


#%%----------------------------------------------------------------------------

elif datentyp == 2:
    
    while read == True:
        
        zähl_ordner = 0
        
        print("Wähle Ordner zum Auswerten:")
        root = Tk()
        root.attributes('-topmost',True)
        root.wm_attributes('-topmost', 1)
        root.withdraw()
        folder = askdirectory(title='Select Folder')
        
        print(folder)
        
        vorlaufszeit_einzeln = float(input("Vorlaufszeit:\n"))
        anstiegszeit_einzeln = float(input("Anstiegszeit:\n"))
        kraft_einzeln = float(input("maximale Kraft:\n"))
        nachlaufzeit_einzeln = float(input("Nachlaufzeit:\n"))
        
        for filename in os.scandir(folder):
            if filename.is_file():
                
                vorlaufszeit = np.append(vorlaufszeit, vorlaufszeit_einzeln)
                anstiegszeit = np.append(anstiegszeit, anstiegszeit_einzeln)
                kraft = np.append(kraft, kraft_einzeln)
                nachlaufzeit = np.append(nachlaufzeit, nachlaufzeit_einzeln)
                
                name, data_read = read_signal(filename, save=True, folder=folder)
                data_filtered = filter(data_read, save=True, folder=folder, name=name)
                max_amplitude_einzeln, energy_einzeln, rise_tim_einzelne, rise_angle_einzeln = features(data_filtered)
                
                time_einzeln = np.array([data_read[:, 0]])
                signal_raw_einzeln = np.array([data_read[:, 1]])
                signal_filtered_einzeln = np.array([data_filtered [:, 1]])
                abstand_einzeln = float(name[-5:-2])
                winkel_einzeln = float(name[0:4])
                
                time = np.vstack((time, time_einzeln))
                signal_raw = np.vstack((signal_raw, signal_raw_einzeln))
                signal_filtered = np.vstack((signal_filtered, signal_filtered_einzeln))
                
                abstand = np.append(abstand, abstand_einzeln)
                winkel = np.append(winkel, winkel_einzeln)
                max_amplitude = np.append(max_amplitude, max_amplitude_einzeln)
                energy = np.append(energy, energy_einzeln)
                
                zähl_ordner = zähl_ordner + 1
                zähl_durchgehend = zähl_durchgehend +1
                
                print(folder, name + str(' done'))
                
        print('Es wurden '+str(zähl_ordner)+' Signale hinzugefügt')
        print('Bisher wurden '+str(zähl_durchgehend)+' Signale eingelesen')
        more_data = input("Sollen mehr Daten eingelesen werden? [y/n]\n")
        
        if more_data == 'y':
            pass
        elif more_data == 'n':
            
            read = False
            
            print("Wähle Ordner zum Abspeichern der hdf5-Datei:")
            root = Tk()
            root.attributes('-topmost',True)
            root.wm_attributes('-topmost', 1)
            root.withdraw()
            folder_save = askdirectory(title='Select Folder')
            print(folder_save)
            
            time = np.delete(time, 0, 0)
            signal_raw = np.delete(signal_raw, 0, 0)
            signal_filtered = np.delete(signal_filtered, 0, 0)
            
            table = pd.DataFrame({"Vorlaufszeit": vorlaufszeit,
                                 "Anstiegszeit": anstiegszeit,
                                 "Kraft": kraft,
                                 "Nachlaufzeit": nachlaufzeit,
                                 "Abstand": abstand,
                                 "Winkel": winkel,
                                 "max_amplitude": max_amplitude,
                                 "energy": energy})
            
            save_name = folder_save+str('\data_3D_')+current_date
            
            with h5py.File(save_name+str(".hdf5"), "a") as f:
                f.create_dataset("time", data = time)
                f.create_dataset("signal_raw", data = signal_raw)
                f.create_dataset("signal_filtered", data = signal_filtered)
                f.create_dataset("table", data=table)
            
            print('Es wurden insgesamt '+str(zähl_durchgehend)+'Signale abgespeichert')