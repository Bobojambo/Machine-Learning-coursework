# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 12:46:01 2018

@author: hakala24
"""



#from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import data
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import glob
from skimage import io
import os

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model

    
def create_CNN():
    
    #N = 512 # Number of feature maps
    #w, h = 5, 5 # Conv. window size
    
    base_model = VGG16(include_top = False, weights = "imagenet",
                  input_shape = (64,64,3))
    
    W = base_model.output
    
    W = Flatten()(W)
    W = Dense(100, activation = 'sigmoid')(W)
    output = Dense(2, activation = 'sigmoid')(W)
    
    model = Model(inputs = [base_model.input], outputs = [output])
    
    model.layers[-7].trainable = True
    model.layers[-6].trainable = True
    model.layers[-5].trainable = True

    
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.summary()
    
    return model



images_data = []
labels = []

class1_folder = 'GTSRB_subset_2/class1/*.jpg'
class2_folder = 'GTSRB_subset_2/class2/*.jpg'

for filename in glob.glob(class1_folder):
    img = image.load_img(filename)
    numpy_img = np.array(img)
    normalized_img = (numpy_img - np.min(numpy_img)) / np.max(numpy_img)
    images_data.append(normalized_img)
    labels.append(0)

for filename in glob.glob(class2_folder):
    img = image.load_img(filename)
    numpy_img = np.array(img)
    normalized_img = (numpy_img - np.min(numpy_img)) / np.max(numpy_img)
    images_data.append(normalized_img)
    labels.append(1)
    
#Koko datalle mielummin normalisointi    

    
labels = np.array(labels)
labels = to_categorical(labels, num_classes = 2)
images_data = np.array(images_data)

X_train, X_test, y_train, y_test = train_test_split(images_data, labels, test_size = 0.2)
    
model = create_CNN() 

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data = [X_test, y_test])