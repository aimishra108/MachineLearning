# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 12:31:14 2020

@author: Atul
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

''' Substitute missing data'''
from sklearn.preprocessing import Imputer
Imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
Imputer=Imputer.fit(X[:,1:3])
X[:,1:3]=Imputer.transform(X[:,1:3])

#Encode categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)


#Splitting set into data set and training set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train.reshape(-1,1))



