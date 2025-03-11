# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:16:53 2023

@author: Simon
"""

import numpy as np
import pandas as pd

file = r"G:\Acoustic Emission\Daten\Lennart\Plate\plate_partial_power_5.csv"

data = pd.read_csv(file)

for i in range(35):
    energy = np.array([])
    for j in range(25):
        energy = np.append(energy, data.iloc[i][-(j+1)])
        energy = energy[~np.isnan(energy)]
    if len(energy) == 25:
        pass
    else:
        dif = 25-len(energy)
        for k in range(dif):
            randoms = np.random.randint(len(energy), size=dif)
            random_value = np.array([])
            for l in range(len(randoms)):
                random_value = np.append(random_value, energy[randoms[l]])
                energy_new = energy
                energy_new = np.append(energy_new, random_value)
            data.iloc[i, -25:] = energy_new

# data.to_csv(r"G:\Acoustic Emission\Daten\Lennart\Plate\plate_partial_power_5.csv", index=False)