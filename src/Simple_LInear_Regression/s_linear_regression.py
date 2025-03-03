# ------------------------------------------------------------------------------
# File: s_linear_regression.py
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

import urllib.request                   # For downloading files from a URL
import matplotlib.pyplot as plt         # For plotting graphs and visualizations
import pandas as pd                     # For handling tabular data
import numpy as np                      # For numerical computations
from sklearn.metrics import r2_score    # For regression performance metrics
from sklearn import linear_model        # For creating a linear regression model

# Define the dataset URL and the local file name to save it
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
filename = "FuelConsumption.csv"

# Download the dataset from the URL
urllib.request.urlretrieve(url, filename)
print("Data obtained:", filename)

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("FuelConsumption.csv")

# Display the first 16 rows of the dataset for inspection
print(df.head(16))

# Display statistical summary of the dataset (e.g., mean, std, min, max, etc.)
print("Statistical Sumary of the dataset")
print(df.describe())

# Select relevant features for the analysis
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print("Sample of relevant features for the analysis")
print(cdf.head(16))  # Display the first 16 rows of the filtered data

# Visualize the data using histograms for each selected feature
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

# Scatter plot: Fuel consumption vs. CO2 emissions
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Combined Fuel Consumption")
plt.ylabel("Emissions")
plt.show()

# Scatter plot: Engine size vs. CO2 emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='green')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# Scatter plot: Number of cylinders vs. CO2 emissions
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red')
plt.xlabel("N° Cylinders")
plt.ylabel("Emissions")
plt.show()

# Split the data into training and testing sets (80% train, 20% test)
msk = np.random.rand(len(df)) < 0.8  # Generate a random mask
train = cdf[msk]  # Training set
test = cdf[~msk]  # Testing set

# Visualize the training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='cyan')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# Create a linear regression model
regr = linear_model.LinearRegression()

# Prepare the training data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Train the model on the training data
regr.fit(train_x, train_y)

# Display the coefficients and intercept of the trained model
print('Coefficients: ', regr.coef_)  # Slope of the regression line
print('Intercept: ', regr.intercept_)  # Y-intercept of the regression line

# Visualize the linear regression model with the training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='magenta')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-k')  # Regression line
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

# Prepare the testing data
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Make predictions using the trained model
y_predictions = regr.predict(test_x)

# Evaluate the model's performance
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predictions - test_y)))  # MAE
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predictions - test_y) ** 2))  # MSE
print("R2-score: %.2f" % r2_score(test_y, y_predictions))  # R2 Score

# Train the model again using FUELCONSUMPTION_COMB as the independent variable
train_x = train[["FUELCONSUMPTION_COMB"]]
test_x = test[["FUELCONSUMPTION_COMB"]]

# Create and train a new regression model
regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

# Make predictions with the new model
predictions = regr.predict(test_x)

# Evaluate the performance of the new model
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(predictions - test_y)))

print("\nProgram Finished\n")