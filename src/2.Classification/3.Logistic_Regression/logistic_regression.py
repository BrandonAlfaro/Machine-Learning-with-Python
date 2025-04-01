# ------------------------------------------------------------------------------
# File: logistic_regression.py
# Description: Implementation of a basic linear regression model in Python.
#
# Author: BrandonAlfaro
# Date: 27/1/2025
#
# License: MIT
# Copyright (c) 2025 BrandonAlfaro
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ------------------------------------------------------------------------------

import pandas as pd                                     # For data manipulation (DataFrames)
import pylab as pl                                      # MATLAB-like interface for plotting and numerical operations
import numpy as np                                      # For numerical operations (arrays, math)
import scipy.optimize as opt                            # For optimization tasks (e.g., equation solving)
from sklearn import preprocessing                       # For data preprocessing (scaling, encoding)
from sklearn.model_selection import train_test_split    # For splitting data into train/test sets
from sklearn.linear_model import LogisticRegression     # For logistic regression modeling
from sklearn.metrics import confusion_matrix            # For evaluating classification models
import matplotlib.pyplot as plt                         # For creating visualizations (graphs, charts)

# Getting dataset
churn_df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv',delimiter=",")
print(churn_df.head())

# Data selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

print(churn_df.shape)

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print(X[0:5])

y = np.asarray(churn_df['churn'])
print(y[0:5])

# Pre-processing data
X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

# Train and test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Modeling the logistic regression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

# Prediction
yhat = LR.predict(X_test)
print(yhat)

# Probability of predictions
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob) # [P(Y=0|X), (Y=1|X)]

# Code Finished
print("\nCode Finished.")