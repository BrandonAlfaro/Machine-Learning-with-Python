# ------------------------------------------------------------------------------
# File: decision_tree.py
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

import pandas as pd                                     # For working with dataframes.
from sklearn.tree import DecisionTreeClassifier         # For creating a decision tree model.
from sklearn import preprocessing                       # For preparing data.
from sklearn.model_selection import train_test_split    # For splitting data into training and testing sets.
from sklearn import metrics                             # For evaluating model performance.

# Load the dataset from an online CSV file.
# The delimiter is a comma, which is standard for CSV files.
my_data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv',delimiter=",")

# Display the first 5 rows of the DataFrame to inspect its contents.
print("\nData about patients.")
print(my_data.head()) # head() is a method.

# Display the size of data
print("\nSize of data (rows, columns).")
print(my_data.shape) # shape is an attribute.

# Pre-processing.
# Splitting the dataset into data and target.
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = my_data['Drug']
# Display variables
print("\nSample of X.")
print(X[0:5])
print("\nSample of y.")
print(y[0:5])

# Convert categorical variables to numerical values.
# Sex variable.
le_sex = preprocessing.LabelEncoder() # Create an instance.
le_sex.fit(['F','M']) # Mapping categorical values to numerical values.
X[:,1] = le_sex.transform(X[:,1]) # Change the categorical values to numerical values.
# Blood Pressure variable
le_BP = preprocessing.LabelEncoder() # Create an instance.
le_BP.fit(['HIGH', 'NORMAL', 'LOW']) # Mapping categorical values to numerical values.
X[:,2] = le_BP.transform(X[:,2]) # Change the categorical values to numerical values.
# Cholesterol variable
le_Chol = preprocessing.LabelEncoder() # Create an instance.
le_Chol.fit([ 'NORMAL', 'HIGH']) # Mapping categorical values to numerical values.
X[:,3] = le_Chol.transform(X[:,3]) # Change the categorical values to numerical values.
# Display the data converted.
print("\nData converted.")
print(X[0:5])

# Setting up the Decision Tree
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3) # Train 70%, test 30%

# Display the size of training and test data.
print("\nShape of X training set {}".format(X_trainset.shape), "shape of y training set {}".format(y_trainset.shape))
print("Shape of X training set {}".format(X_testset.shape), "shape of y training set {}".format(y_testset.shape))

# Setting up the model of Decision Tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print("\nDefault parameters of the Decision Tree model.")
print(drugTree.get_params()) # Shows the default parameters

drugTree.fit(X_trainset,y_trainset) # The method learns patterns from the training dataset.

# Prediction
predTree = drugTree.predict(X_testset) # The method makes preditions for the test dataset.

# Display the fist 5 preditions and real results.
print("\nPredictions of model.")
print (predTree [0:5])
print("Real values of data.")
print (y_testset [0:5])

# Evaluation
print("\nDecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# End of code
print("\nCode Finished.\n")