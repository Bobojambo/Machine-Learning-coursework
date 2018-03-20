# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:31:46 2018

@author: hakala24
"""

import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cross_validation import train_test_split
from skimage import data
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier




#GTSRB data


class1_images = []
class2_images = []
numPoints = 24
radius = 3
truelabels = []
hist_data = []

for x in range(0,100):
    no = str(x).zfill(3)
    img1 = data.load("C:/Users/hakala24/SpyderProjects/SGN_course/GTSRB_subset/class1/" + no + ".jpg")
    img2 = data.load("C:/Users/hakala24/SpyderProjects/SGN_course/GTSRB_subset/class2/" + no + ".jpg")
    lbp1 = local_binary_pattern(img1, numPoints,radius, method="uniform")
    lbp2 = local_binary_pattern(img2, numPoints,radius, method="uniform")
    hist_data.append(np.histogram(lbp1)[0])
    truelabels.append(1)
    hist_data.append(np.histogram(lbp2)[0])
    truelabels.append(2)


X_train, X_test, y_train, y_test = train_test_split(hist_data, truelabels, test_size=0.2, random_state = 13245)

#Classifiers
clf_list = [LogisticRegression(), SVC()]
clf_name = ["LR", "SVC"]

#Create C hyperparameter values
C_range = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1]

Normalized = False

for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            if name == "SVC" and Normalized == False:
                print("normalize data")
                normalizer = Normalizer()
                X_train = normalizer.fit_transform(X_train, y_train)
                X_test = normalizer.fit_transform(X_test, y_test)
                Normalized == True
                
                
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(name, " Score: ", score)


X_train, X_test, y_train, y_test = train_test_split(hist_data, truelabels, test_size=0.2, random_state = 13245)
#Task 5
n_trees = 100
clf_list2 = [RandomForestClassifier(n_estimators = n_trees), ExtraTreesClassifier(n_estimators = n_trees), GradientBoostingClassifier(n_estimators = n_trees), AdaBoostClassifier(n_estimators = n_trees) ]
clf_name2 = ["Random Forest", "Extra Trees", "GradientBoost", "Adaboost"]
for clf,name in zip(clf_list2, clf_name2):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(name, " Score: ", score)
    
    
