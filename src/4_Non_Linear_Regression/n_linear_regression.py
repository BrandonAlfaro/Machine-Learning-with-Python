# ------------------------------------------------------------------------------
# File: n_linear_regression.py
# Description: Implementation of a basic non-linear regression model in Python.
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
from scipy.optimize import curve_fit    # For fitting functions to data using optimization

# Define the dataset URL and the local file name to save it
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv"
filename = "China_GDP.csv"

# Download the dataset from the URL
urllib.request.urlretrieve(url, filename)
print("\nData obtained:", filename)

# Load the dataset into a Pandas DataFrame
df = pd.read_csv("China_GDP.csv")
df.head(10)


## Linear regression example with generate data.
x = np.arange(-5.0, 5.0, 0.1)
y = 2*(x) + 3                                     # Linear equation
y_noise = 2 * np.random.normal(size=x.size)       # Noise generate
ydata = y + y_noise                               # Equivalent data
# Plot the data and linear equation
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


## Polynomial regression example with generate data.
x = np.arange(-5.0, 5.0, 0.1)
y = 1*(x**3) + 1*(x**2) + 1*x + 3                 # Polynomial equation(degree 3)
y_noise = 20 * np.random.normal(size=x.size)      # Noise generate
ydata = y + y_noise                               # Equivalent data
# Plot the data and polynomial equation
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


## Quadratic regression example with generate data.
x = np.arange(-5.0, 5.0, 0.1)
y = np.power(x,2)                                 # Quadratic equation
y_noise = 2 * np.random.normal(size=x.size)       # Noise generate
ydata = y + y_noise                               # Equivalent data
# Plot the data and quadratic equation
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


## Exponential regression example with generate data.
x = np.arange(-5.0, 5.0, 0.1)
y= np.exp(x)                                      # Exponential equation
y_noise = 2 * np.random.normal(size=x.size)       # Noise generate
ydata = y + y_noise                               # Equivalent data
# Plot the data and exponential equation
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


## Logarithmic regression example with generate data.
x = np.arange(-5.0, 5.0, 0.1)
y = np.log(x)                                     # Logarithmic equation
y_noise = 0.3 * np.random.normal(size=x.size)       # Noise generate
ydata = y + y_noise                               # Equivalent data
# Plot the data and logarithmic equation
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


# Sigmoidal/Logistic
x = np.arange(-5.0, 5.0, 0.1)
y = 1-4/(1+np.power(3, x-2))                      # Sigmoidal/Logistic
y_noise = 0.5 * np.random.normal(size=x.size)       # Noise generate
ydata = y + y_noise                               # Equivalent data
# Plot the data and logarithmic equation
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()


## Dataset from China's GDP
x_data, y_data = (df["Year"].values, df["Value"].values)
# Plot China's GDP dataset
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


## Choosing building a model
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

# Parameters of model(By eye)
beta_1 = 0.10
beta_2 = 1990.0

# Sigmoidal/Logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

# Plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()

# Data normalization
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# Fit parameters and parameters covariance
popt, pcov = curve_fit(sigmoid, xdata, ydata)
# Print the best parameters
print("\nbeta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Print the regression model with normalized data
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# Build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# Evaluation of model selected
print("\nMean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,y_hat) )

print("\nProgram Finished\n")