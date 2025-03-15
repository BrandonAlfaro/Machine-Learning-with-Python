# ------------------------------------------------------------------------------
# File: s_linear_regressiok_nearest_neigbors.py
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

import numpy as np                                      # Import NumPy for numerical operations
import matplotlib.pyplot as plt                         # Import Matplotlib for data visualization
import pandas as pd                                     # Import Pandas for data manipulation
from sklearn import preprocessing                       # Import preprocessing tools from scikit-learn
from sklearn.model_selection import train_test_split    # Import function to split data into training and test sets
from sklearn.neighbors import KNeighborsClassifier      # Import KNN classifier
from sklearn import metrics                             # Import metrics to evaluate model performance

# Load dataset from a URL into a Pandas DataFrame
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()  # Display the first 5 rows of the dataset

# Display column names
print("\nDisplay column names.")
print(df.columns)

# Count the occurrences of each category in the 'custcat' column
print("\nOccurrences of each category in 'custcat' column.")
print(df['custcat'].value_counts())

# Plot a histogram of the 'income' column with 50 bins
df.hist(column='income', bins=50)
plt.title("Histogram of 'income'")
plt.show()

# Extract feature columns and convert them into a NumPy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  
print("\nFirst 5 rows of independent variables of dataset.")
print(X[0:5])  # Display the first 5 rows of the feature matrix

# Extract the target variable and convert it into a NumPy array
print("\nFirst 5 values of dependent variables of dataset.")
y = df['custcat'].values
print(y[0:5])  # Display the first 5 values of the target variable

# Standardize the features (zero mean and unit variance)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print("\nStandardize data. With X' = (X - μ) / σ")
print("First 5 rows after standardization.")
print(X[0:5])  # Display the first 5 rows after standardization

# Split the dataset into training (80%) and testing (20%) sets
print("\nSplit data set into training and testing(80/20).")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)  # Print training set dimensions
print('Test set:', X_test.shape, y_test.shape)  # Print test set dimensions

# Test K-NN

# Define the number of neighbors for 4-NN
print("\n4 Nearest Neighbors.")
k = 4
# Train the 4-NN model
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# Make predictions using the test set
yhat = neigh.predict(X_test)
print("First 5 predictions.")
print(yhat[0:5])  # Display the first 5 predictions
print("First 5 real values.")
print(y_test[0:5])  # Display the first 5 predictions
# Evaluate model accuracy on training and test sets
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Define the number of neighbors for 6-NN
print("\n6 Nearest Neighbors.")
k = 6
# Train the 4-NN model
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# Make predictions using the test set
yhat = neigh.predict(X_test)
print("First 5 predictions.")
print(yhat[0:5])  # Display the first 5 predictions
print("First 5 real values.")
print(y_test[0:5])  # Display the first 5 predictions
# Evaluate model accuracy on training and test sets
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Define the number of neighbors for 2-NN
print("\n2 Nearest Neighbors.")
k = 2
# Train the 4-NN model
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
# Make predictions using the test set
yhat = neigh.predict(X_test)
print("First 5 predictions.")
print(yhat[0:5])  # Display the first 5 predictions
print("First 5 real values.")
print(y_test[0:5])  # Display the first 5 predictions
# Evaluate model accuracy on training and test sets
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# Accuracy of K-NN for different values of K
Ks = 10  # Define the maximum number of neighbors to test

# Create zero arrays to store the mean accuracy and standard deviation for each KNN model
mean_acc = np.zeros((Ks-1))  # Mean accuracy for each k value
std_acc = np.zeros((Ks-1))   # Standard deviation of accuracy for each k

# Loop to test different k values in the K-NN model
for n in range(1, Ks):  # Iterate from k=1 to k=9
    
    # Train the KNN model with the current number of neighbors (n)
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    
    # Make predictions on the test set
    yhat = neigh.predict(X_test)
    
    # Compute the model's accuracy and store it in the mean_acc array
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    # Compute the standard deviation of accuracy and store it in std_acc
    std_acc[n-1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Display the array containing mean accuracy for each k value
print("\nMean test set accuracy for each value.")
print(mean_acc)

# Plot the accuracy for each K value
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.title("Model accuracy")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()