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

# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn  # Overrides the warning function to prevent warnings from being displayed.

# Import necessary libraries
import sys  # Not used in this code, could be removed if unnecessary.
import numpy as np  # Typically used for numerical computations, but not yet used here.
import pandas as pd  # Used for handling and manipulating data in DataFrames.
from sklearn.tree import DecisionTreeClassifier  # Imports the decision tree classifier from scikit-learn.
import sklearn.tree as tree  # Imports the tree module (seems redundant with the previous line).

# Load the dataset from an online CSV file.
# The delimiter is a comma, which is standard for CSV files.
my_data = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv',
    delimiter=","
)

# Display the first 5 rows of the DataFrame to inspect its contents.
my_data.head()
