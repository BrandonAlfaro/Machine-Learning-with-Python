# ------------------------------------------------------------------------------
# File: multiple_linear_regression.py
# Description: Implementation of a multiple linear regression model in Python.
#
# Author: Brandon Alfaro
# Date: 3/3/2025
#
# License: MIT
# Copyright (c) 2025 Brandon
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

import urllib.request                   # For downloading files from a URL
import matplotlib.pyplot as plt         # For plotting graphs and visualizations
import pandas as pd                     # For handling tabular data
import numpy as np                      # For numerical computations
from sklearn import linear_model        # For creating a linear regression model
from sklearn.metrics import r2_score    # For evaluating regression models

# Define the dataset URL and the local file name to save it
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
filename = "FuelConsumption.csv"

# Download the dataset from the URL
urllib.request.urlretrieve(url, filename)
print("Data obtained:", filename)

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(filename)

# Display the first 9 rows of the dataset for inspection
print("Sample of dataset:")
print(df.head(9))

# Select relevant features for analysis
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print("Sample of selected features:")
print(cdf.head(9))

# Scatter plot: Engine size vs. CO2 emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# Split the data into training and testing sets (80% train, 20% test)
msk = np.random.rand(len(df)) < 0.8  # Generate a random mask
train = cdf[msk]  # Training set
test = cdf[~msk]  # Testing set

# Scatter plot: Training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='green')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# Create and train the first regression model
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)

# Display the coefficients of the trained model
print('Coefficients:', regr.coef_)

# Make predictions on the test set
y_pred = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

# Evaluate the model
print("Multiple linear regression with engine size, cylinders and flue consumption for C02 emissions.")
print("Mean Squared Error (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("Variance score (R2): %.2f\n" % r2_score(y_test, y_pred))

# Train and evaluate a second regression model using different features
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)
print('Coefficients:', regr.coef_)

# Make predictions
y_pred = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

# Evaluate the second model
print("Multiple linear regression with engine size, cylinders, fuel consumption in city and fuel consumption HWY for C02 emissions.")
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test) ** 2))
print("Variance score (R2): %.2f" % r2_score(y_test, y_pred))

print("\nProgram Finished\n")
