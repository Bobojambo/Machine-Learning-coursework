# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:15:59 2018

@author: hakala24
"""

from PIL import Image
import glob

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
#from skimage.feature import local_binary_pattern
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.svc import SVC
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from skimage import data
from sklearn.metrics import accuracy_score

#Task 3---------------------------------------------------------------------

class1_folder = 'GTSRB_subset/class1/*.jpg'
class2_folder = 'GTSRB_subset/class2/*.jpg'

class1_images = []
class2_images = []
numPoints = 24
radius = 3
truelabels = []
hist_data = []

#for file in glob.glob(class1_folder):    
#    class1_images.append(data.load(file))
#    
#for file in glob.glob(class2_folder):
#    class2_images.append(data.load(file))

for x in range(0,100):
    no = str(x).zfill(3)
    print(no)
    img1 = data.load("C:/Users/hakala24/SpyderProjects/SGN_course/GTSRB_subset/class1/" + no + ".jpg")
    img2 = data.load("C:/Users/hakala24/SpyderProjects/SGN_course/GTSRB_subset/class2/" + no + ".jpg")
    lbp1 = local_binary_pattern(img1, numPoints,radius, method="uniform")
    lbp2 = local_binary_pattern(img2, numPoints,radius, method="uniform")
    hist_data.append(np.histogram(lbp1)[0])
    truelabels.append(1)
    hist_data.append(np.histogram(lbp2)[0])
    truelabels.append(2)
    
#(hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))

# normalize the histogram
#hist = hist.astype("float")
#hist /= (hist.sum() + 1e-7)

#plt.hist(hist, bins='auto')

#Task 4-----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(hist_data[:], truelabels, test_size=0.2, random_state = 42)

models = []
clf = KNeighborsClassifier()
models.append(["KNN",clf])
clf = LinearDiscriminantAnalysis()
models.append(["LDA",clf])
clf = SVC()
models.append(["SVC",clf])
 
for modelrow in models:
    print(modelrow[0])
    model = modelrow[1]
    #for transformed_data_row in transformed_data:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
        #length = len(transformed_data_row[0])
        #emptyspace = "                        "
        #for i in range(length):
        #    emptyspace = emptyspace[:-1]
    print("      Accuracy: ", accuracy)

#Task 5-----------------------------------------------------------------------

def gaussian(x, mu, sigma):
    
    first_term = 1/np.sqrt(2*np.pi*sigma*sigma)
    exp_term = -1/(2*sigma*sigma)*(x-mu)*(x-mu)
    p = first_term * np.exp(exp_term)
    return p


def log_gaussian(x, mu, sigma):
    
    p = np.log(1/np.sqrt(2*np.pi*sigma**2)) - 1/(2*sigma**2)*(x-mu)**2
    return p

x = np.linspace(-5, 5, num=50)
plt.figure(1)
plt.plot(x,gaussian(x,0,1))
plt.figure(2)
plt.plot(x,log_gaussian(x,0,1))

