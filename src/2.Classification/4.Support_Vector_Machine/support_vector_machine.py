# === Library Imports ===
# Import essential libraries for data manipulation, visualization, and model training

from sklearn.model_selection import train_test_split    # For splitting data into train and test sets
from sklearn import svm                                 # Import the Support Vector Machine module from scikit-learn
from sklearn.metrics import classification_report       # Detailed precision/recall/F1 report for each class
from sklearn.metrics import confusion_matrix            # Matrix of true vs predicted classifications (TP/FP/TN/FN)
from sklearn.metrics import f1_score                    # Harmonic mean of precision and recall (class balance metric)
from sklearn.metrics import jaccard_score               # Similarity score measuring intersection-over-union of sets
import pandas as pd                                     # For data handling and DataFrame operations
import numpy as np                                      # For numerical computing and array manipulation
import matplotlib.pyplot as plt                         # For data visualization
import pylab as pl                                      # Alternative plotting library (rarely used directly)
import scipy.optimize as opt                            # For optimization algorithms (not used yet in this script)
import requests                                         # Downloading data/files from the internet
import itertools                                        # Import utility for looping over matrix coordinates
import numpy as np                                      # Import NumPy for numerical operations
import matplotlib.pyplot as plt                         # Import Matplotlib for plotting

# === Define Helper Function to Plot Confusion Matrix ===
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    If 'normalize' is True, it shows percentages instead of counts.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)  # Print the matrix values to the console

    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # Display the matrix as an image
    plt.title(title)  # Set the title of the plot
    plt.colorbar()  # Show color scale

    tick_marks = np.arange(len(classes))  # Define positions for axis ticks
    plt.xticks(tick_marks, classes, rotation=45)  # Set X-axis labels and rotate for readability
    plt.yticks(tick_marks, classes)  # Set Y-axis labels

    fmt = '.2f' if normalize else 'd'  # Format values as float or integer depending on normalization
    thresh = cm.max() / 2.  # Define a threshold to choose text color based on value brightness

    # Annotate each cell with its numeric value
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.ylabel('True label')  # Label Y-axis
    plt.xlabel('Predicted label')  # Label X-axis

    plt.show()

# === Download Dataset ===
# Define the URL where the CSV dataset is hosted
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"
response = requests.get(url)                            # Send an HTTP GET request to download the file content from the specified URL
# Create a local file in binary write mode and write the downloaded content into it
with open("cell_samples.csv", "wb") as f:
    f.write(response.content)


# === Load Dataset ===
cell_df = pd.read_csv("cell_samples.csv")               # Read CSV file into a pandas DataFrame

# Preview the first few rows to understand the structure and contents
print(cell_df.head(), "\n")


# === Data Visualization ===
# Plot two subsets of the data to visually explore feature relationships
ax = cell_df[cell_df['Class'] == 4][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant'
)
cell_df[cell_df['Class'] == 2][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax
)

# Show the combined scatter plot
plt.show()


# === Data Cleaning ===
# Check data types of each column
print(cell_df.dtypes, "\n")

# Convert 'BareNuc' column to numeric, coercing errors to NaN, and filter out invalid entries
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()] # Creates a boolean mask to keep the numbers
cell_df['BareNuc'] = cell_df['BareNuc'].astype(int) # Remaining values to integers

# Confirm the changes in data types
print(cell_df.dtypes, "\n")


# === Feature Selection ===
# Select relevant features to be used as input variables (X)
feature_df = cell_df[[
    'Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
    'BareNuc', 'BlandChrom', 'NormNucl', 'Mit'
]]

# Convert selected features to a NumPy array
X = np.asarray(feature_df)
print(X[0:5], "\n")  # Inspect the first 5 feature rows

# Define the target variable y
y = np.asarray(cell_df['Class'])
print(y[0:5],"\n")  # Inspect the first 5 values


# === Split Data into Training and Test Sets ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)    # random_state ensures reproducibility, 80% training, 20% testing

# Display the shape of the resulting datasets
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# === Train SVM Model with RBF Kernel ===
clf = svm.SVC(kernel='rbf')  # Create an SVM classifier instance using the RBF (non-linear) kernel
clf.fit(X_train, y_train)  # Train the model using the training data
yhat = clf.predict(X_test)  # Predict the test set labels using the trained model
print(yhat[0:5], "\n")  # Display the first 5 predictions for a quick sanity check

# Print evaluation metrics for comparison
print("\n--- No-Linear Kernel Evaluation ---")  # Section separator
print("F1-score (weighted): %.4f" % f1_score(y_test, yhat, average='weighted'))
print("Jaccard score (label=2): %.4f" % jaccard_score(y_test, yhat, pos_label=2))

# === Evaluate RBF Kernel Model ===
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2, 4])  # Compute confusion matrix for class labels 2 and 4
np.set_printoptions(precision=2)  # Set NumPy print precision for better readability

plt.figure()  # Create a new figure for the confusion matrix plot
plot_confusion_matrix(
    cnf_matrix,                     # Pass the confusion matrix
    classes=['Benign(2)', 'Malignant(4)'],  # Define class names
    normalize=False,               # Use raw counts instead of normalized values
    title='Confusion Matrix - RBF Kernel'  # Set plot title
)

print(classification_report(y_test, yhat))  # Print precision, recall, f1-score and support

# === Retrain Model Using a Linear Kernel ===

clf2 = svm.SVC(kernel='linear')  # Create a new SVM classifier with a linear kernel
clf2.fit(X_train, y_train)  # Train the linear model on the same training data
yhat2 = clf2.predict(X_test)  # Predict the test set labels using the linear model

# Print evaluation metrics for comparison
print("\n--- Linear Kernel Evaluation ---")  # Section separator
print("F1-score (weighted): %.4f" % f1_score(y_test, yhat2, average='weighted'))  # F1-score for linear kernel
print("Jaccard score (label=2): %.4f" % jaccard_score(y_test, yhat2, pos_label=2))  # Jaccard index for label 2
