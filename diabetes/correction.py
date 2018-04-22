#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:20:32 2018

@author: srinivas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('diabetes.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,8].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
   
#training set 
X1=X_train[:,:1]      #Pregnancy
X2=X_train[:,1:2]     #Glucose
X3=X_train[:,2:3]     #Blood pressure
X4=X_train[:,3:4]     #Skin thickness
X5=X_train[:,4:5]     #Insulin
X6=X_train[:,5:6]     #BMI
X7=X_train[:,6:7]     #Pedigree function
X8=X_train[:,7:8]     #Age
pred1=X1[:,:]
pred1=np.append(pred1,X2,axis=1)
pred1=np.append(pred1,X4,axis=1)
pred1=np.append(pred1,X5,axis=1)
pred1=np.append(pred1,X6,axis=1)
pred1=np.append(pred1,X7,axis=1)
pred1=np.append(pred1,X8,axis=1)
#test set
X_1=X_test[:,:1]      #Pregnancy
X_2=X_test[:,1:2]     #Glucose
X_3=X_test[:,2:3]     #Blood pressure
X_4=X_test[:,3:4]     #Skin thickness
X_5=X_test[:,4:5]     #Insulin
X_6=X_test[:,5:6]     #BMI
X_7=X_test[:,6:7]     #Pedigree function
X_8=X_test[:,7:8]     #Age

pred_1=X_1[:,:]
pred_1=np.append(pred_1,X_2,axis=1)
pred_1=np.append(pred_1,X_4,axis=1)
pred_1=np.append(pred_1,X_5,axis=1)
pred_1=np.append(pred_1,X_6,axis=1)
pred_1=np.append(pred_1,X_7,axis=1)
pred_1=np.append(pred_1,X_8,axis=1)

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(pred1,y_train)

y_pred=classifier.predict(pred_1)

from sklearn.metrics import confusion_matrix
cm_forest = confusion_matrix(y_test, y_pred)








