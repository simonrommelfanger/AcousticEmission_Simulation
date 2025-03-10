# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 20:23:03 2023

@author: Simon
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file_sim = r"G:\Acoustic Emission\Daten\Rohdaten\Platte_Voigt_Notation\evaluated\features_all.csv"
file_real = r"G:\Acoustic Emission\Daten\Lennart\Plate\evaluated\features_all.csv"
file_real_mean = r"G:\Acoustic Emission\Daten\Lennart\Plate\evaluated\features_all_mean.csv"

data_sim = pd.read_csv(file_sim)
data_real = pd.read_csv(file_real)
data_real_mean = pd.read_csv(file_real_mean)

data_sim_features = data_sim.drop('angle', axis=1)
data_sim_features = data_sim_features.drop('radius', axis=1)
data_real_features = data_real.drop('angle', axis=1)
data_real_features = data_real_features.drop('radius', axis=1)
data_real_mean_features = data_real_mean.drop('angle', axis=1)
data_real_mean_features = data_real_mean_features.drop('radius', axis=1)

corr_sim = data_sim_features.corr()
corr_real = data_real_features.corr()
corr_real_mean = data_real_mean_features.corr()

# corr_sim.to_csv(r"G:\Acoustic Emission\Correlation\corr_sim.csv")
# corr_real.to_csv(r"G:\Acoustic Emission\Correlation\corr_real.csv")
# corr_real_mean.to_csv(r"G:\Acoustic Emission\Correlation\corr_real_mean.csv")

x = 'rise_angle'
y = 'energy_fft'

# plt.figure(figsize=[13,10])
# plt.plot(data_sim[x], data_sim[y], 'ok')
# plt.grid('on')
# plt.show()


mask = np.zeros_like(corr_real_mean)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(15, 17))                        
    ax = sns.heatmap(corr_real_mean, annot = True, mask=mask, square=True,
                     cmap='coolwarm', linewidths=2)
plt.tick_params(labelsize = 15) 
plt.show()