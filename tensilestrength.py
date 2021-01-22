# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:41:43 2019

@author: Aman Shukla
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('tensile.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('most fitting line(Linear Regression)')
    plt.xlabel('oven residence time')
    plt.ylabel('tensile strength')
    plt.show()
    return
viz_linear()
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)
# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('most fitting curve(multiple Regression)')
    plt.xlabel('oven residence time')
    plt.ylabel('tensile strength')
    plt.show()
    return
viz_polymonial()
# Predicting a new result with Linear Regression
lin_reg.predict([[40]])
#output 
# Predicting a new result with Polymonial Regression
pol_reg.predict(poly_reg.fit_transform([[40]]))
#output
