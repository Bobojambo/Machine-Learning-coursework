# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:18:38 2018

@author: hakala24
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Exercise 1

filename = "locationData.csv"
array = np.loadtxt(filename)
print(array.shape)
print(array)


#Exercise 2

two_d_plot = plt.plot(array[:,0], array[:,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(array[:,0], array[:,1], array[:,2])



#Exercise 3
def normalize_data_example(data):
    return (data - np.mean(data, axis= 0)) / np.std(data, axis=0)

def normalize_data(X):
    
    print(np.mean(X))
    print(np.std(X))

    X_column1 = np.array(X[:,0])
    X_column2 = np.array(X[:,1])
    X_column3 = np.array(X[:,2])
    
    X_norm_column1 = ((X_column1 - X_column1.mean() )/ X_column1.std())
    X_norm_column2 = ((X_column2 - X_column2.mean() )/ X_column2.std())
    X_norm_column3 = ((X_column3 - X_column3.mean() )/ X_column3.std())

    X_norm = np.array([X_norm_column1, X_norm_column2, X_norm_column3])
    X_norm = X_norm.transpose()        
    
    return X_norm



