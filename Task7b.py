# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:33:29 2018

@author: hakala24
"""

from sklearn.feature_selection import RFECV
import numpy as np
from scipy import io
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = io.loadmat('arcene.mat')

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train'].ravel()
y_test = data['y_test'].ravel()

rfe = RFECV(estimator=LogisticRegression(), step=50)#, verbose=1)
rfe.fit(X_train, y_train)

print("Number of features selected: ", rfe.support_)

print("score: ",rfe.score(X_test, y_test))

plt.plot(range(0,10001,50), rfe.grid_scores_)

#Task 5

L1clf = LogisticRegression(penalty = 'l1')
C_values = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10, 100, 1000]
C_values = np.logspace(-6, 3, 20)

i = 0
for value in C_values:
    clf = LogisticRegression(penalty = 'l1', C = C_values[i])
    clf.fit(X_train, y_train)
    print("C value: ", C_values[i], " Score: ", clf.score(X_test, y_test), " Number of features: ", clf.coef_)
    
    i += 1