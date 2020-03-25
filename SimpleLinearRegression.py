# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:55:07 2020

@author: Atul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_regr=pd.read_csv('Salary_Data.csv')
X=dataset_regr.iloc[:,:-1].values
Y=dataset_regr.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Fitting SLR to Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Prediction
y_pred=regressor.predict(X_test)

#Visualization for Training Set
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualization for Test Set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()