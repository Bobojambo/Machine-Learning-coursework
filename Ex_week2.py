# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:51:26 2018

@author: hakala24
"""

#Exercise 2


import glob
import numpy as np
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.image import imread

#Task 2

#files = glob.glob('leastSquaresData/*.npy')

#for file in files:
#    print(file)
    
x = np.load('leastSquaresData/x.npy')
y = np.load('leastSquaresData/y.npy')

sum_x = np.sum(x)
sum_y = np.sum(y)

sum_xy = np.sum(x*y)

sum_x2 = np.sum(x*x)

x_average = sum_x / x.size
y_average = sum_y / y.size

n = x.size

b = (n*sum_xy - (sum_x * sum_y)) / (n * sum_x2 - math.pow(sum_x, 2))
a = 1/n * sum_y - b * 1/n * sum_x

print("y = ",b ,"x +", a)

#Task3
with open("Ex2_data/locationData.csv", "r") as csv_file:
    
    data = []
    for line in csv_file:
        values = line.split(" ")
        values = values[:]
        values = [float(v) for v in values]
        data.append(values)
        
data_numpy = np.loadtxt("Ex2_data/locationData.csv")
np.all(data==data_numpy)
np.any()
np.all()



#Task 4
mat = loadmat("Ex2_data/twoClassData.mat")
print(mat.keys()) # Which variables mat contains?
X = mat["X"] # Collect the two variables.
y = mat["y"].ravel()

#Kuinka tarkastaa labelit paremmin
#plt.plot(X[y==0, :], 'bo')
plt.plot(X[y==0,0], X[y==0,1], 'bo')
plt.plot(X[y==1,0], X[y==1,1], 'ro')
plt.plot(X[:,0], X[:,1], 'go')
#plt.plot(X[200:, 0], X[200:, 1], 'ro')

plt.plot()

#Task 5

# Read the data

img = imread("Ex2_data/uneven_illumination.jpg")
plt.imshow(img, cmap='gray')
plt.title("Image shape is %dx%d" % (img.shape[1], img.shape[0]))
plt.show()

# Create the X-Y coordinate pairs in a matrix
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel()
z = Z.ravel()


# ********* TODO 1 **********
# Create data matrix
# Use function "np.column_stack".
# Function "np.ones_like" creates a vector like the input.
H = np.column_stack((x*x, y*y, x*y, x, y, np.ones_like(x)))



# ********* TODO 2 **********
# Solve coefficients
# Use np.linalg.lstsq
# Put coefficients to variable "theta" which we use below.
theta = np.linalg.lstsq(H,z)[0]


# Predict
z_pred = H @ theta
Z_pred = np.reshape(z_pred, X.shape)

# Subtract & show
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()

