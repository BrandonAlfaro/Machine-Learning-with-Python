import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
import pandas as pd  # Import Pandas for data manipulation
from sklearn import preprocessing  # Import preprocessing tools from scikit-learn
from sklearn.model_selection import train_test_split  # Import function to split data into training and test sets
from sklearn.neighbors import KNeighborsClassifier  # Import KNN classifier
from sklearn import metrics  # Import metrics to evaluate model performance

# Load dataset from a URL into a Pandas DataFrame
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()  # Display the first 5 rows of the dataset

# Count the occurrences of each category in the 'custcat' column
df['custcat'].value_counts()

# Plot a histogram of the 'income' column with 50 bins
df.hist(column='income', bins=50)

# Display column names
print(df.columns)

# Extract feature columns and convert them into a NumPy array
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  
X[0:5]  # Display the first 5 rows of the feature matrix

# Extract the target variable and convert it into a NumPy array
y = df['custcat'].values
y[0:5]  # Display the first 5 values of the target variable

# Standardize the features (zero mean and unit variance)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]  # Display the first 5 rows after standardization

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)  # Print training set dimensions
print('Test set:', X_test.shape, y_test.shape)  # Print test set dimensions

# Define the number of neighbors for KNN
k = 4  
# Train the KNN model
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
neigh  # Display the trained model

# Make predictions using the test set
yhat = neigh.predict(X_test)
yhat[0:5]  # Display the first 5 predictions

# Evaluate model accuracy on training and test sets
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
