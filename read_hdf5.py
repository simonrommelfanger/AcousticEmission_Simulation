# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:34:50 2023

@author: Simon
"""

import h5py
import matplotlib.pyplot as plt
import pandas as pd

with h5py.File(r"D:\Downloads\data_2D_27032023.hdf5", 'r') as f:
    time = f['time'][()]
    signal_raw = f['signal_raw'][()]
    signal_filtered = f['signal_filtered'][()]
    table_array = f['table'][()]
    keys = list(f.keys())

table = pd.DataFrame({'Vorlaufzeit': table_array[:, 0],
                      'Anstiegszeit': table_array[:, 1],
                      'Kraft': table_array[:, 2],
                      'Nachlaufzeit': table_array[:, 3],
                      'Abstand': table_array[:, 4],
                      'Winkel': table_array[:, 5],
                      'max. Amplitude': table_array[:, 6],
                      'Energie': table_array[:, 7]})

for i in range(time.shape[0]):
    plt.figure(figsize=[13,10])
    plt.plot(time[i], signal_filtered[i])
    plt.show()
    print(i)