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
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# Getting dataset
churn_df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv',delimiter=",")  # Load dataset from URL
print(churn_df.head())  # Display first 5 rows of the dataframe

# Data selection
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]  # Select specific columns
churn_df['churn'] = churn_df['churn'].astype('int')  # Convert 'churn' column to integer type
print(churn_df.head())  # Display first 5 rows of modified dataframe

print(churn_df.shape)  # Print dimensions of the dataframe (rows, columns)

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])  # Create feature matrix from selected columns
print(X[0:5])  # Print first 5 rows of feature matrix

y = np.asarray(churn_df['churn'])  # Create target vector from 'churn' column
print(y[0:5])  # Print first 5 values of target vector

# Pre-processing data
X = preprocessing.StandardScaler().fit(X).transform(X)  # Standardize features (mean=0, variance=1)
print(X[0:5])  # Print first 5 rows of standardized features

# Train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)  # Split data into 80% train, 20% test
print('Train set:', X_train.shape, y_train.shape)  # Print dimensions of training set
print('Test set:', X_test.shape, y_test.shape)  # Print dimensions of test set

# Modeling the logistic regression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)  # Create and train logistic regression model
print(LR)  # Print model parameters

# Prediction
yhat = LR.predict(X_test)  # Make predictions on test set
print(yhat)  # Print predicted class labels

# Probability of predictions
yhat_prob = LR.predict_proba(X_test)  # Calculate class probabilities for test set
print(yhat_prob)  # Print probabilities [P(Y=0|X), P(Y=1|X)] for each sample

# Evaluation
jaccard_score(y_test, yhat,pos_label=0)


# Code Finished
print("\nCode Finished.")  # Indicate end of script