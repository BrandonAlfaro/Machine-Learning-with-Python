# ------------------------------------------------------------------------------
# File: k-means.py
# Description: Implementation of a k-means clustering model in Python.
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

# Importing essential libraries
from sklearn.preprocessing import StandardScaler    # For standardizing features
from mpl_toolkits.mplot3d import Axes3D             # For 3D plotting capabilities
from sklearn.cluster import KMeans                  # Import KMeans clustering algorithm from scikit-learn
from sklearn.datasets import make_blobs             # Utility to generate sample data (blobs/clusters)
import numpy as np                                  # NumPy: library for numerical computations, especially with arrays
import matplotlib.pyplot as plt                     # Matplotlib: used for creating static visualizations/plots
import pandas as pd                                 # Pandas for data manipulation and analysis
import warnings                                     # For handling warnings

# Disable warnings for cleaner output
def warn(*args, **kwargs):
    pass

# Override the default warning system to suppress all warnings
warnings.warn = warn

# ==============================================
# EXAMPLE 1: Synthetic Data Clustering
# ==============================================

# Set a random seed to ensure reproducible results
np.random.seed(0)

# Generate synthetic clustered data:
X, y = make_blobs(
    n_samples=5000,                      # Total number of data points
    centers=[[4,4], [-2, -1], [2, -3], [1, 1]],  # Coordinates of cluster centers
    cluster_std=1                        # Spread of each cluster
)

# Create scatter plot of the raw data
plt.scatter(X[:, 0], X[:, 1], marker='.')  # Plot all points with small dots
plt.show()  # Display the plot

# Initialize KMeans clustering model with parameters:
k_means = KMeans(
    init="k-means++",   # Smart initialization method for centroids
    n_clusters=4,       # Number of clusters we want to find
    n_init=12           # Number of times algorithm runs with different centroids
)

# Train the KMeans model on our data
k_means.fit(X)

# Get results from the trained model:
k_means_labels = k_means.labels_          # Cluster labels for each point
k_means_cluster_centers = k_means.cluster_centers_  # Coordinates of cluster centers

# Create a figure to visualize the clustering results
fig = plt.figure(figsize=(6, 4))  # Set figure size to 6x4 inches

# Generate a color spectrum for the clusters
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Add a single subplot to our figure
ax = fig.add_subplot(1, 1, 1)

# Plot each cluster with its assigned color
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    # Create a mask for points in the current cluster
    my_members = (k_means_labels == k)
    
    # Get the center of this cluster
    cluster_center = k_means_cluster_centers[k]
    
    # Plot all points in this cluster
    ax.plot(
        X[my_members, 0], X[my_members, 1], 'w',  # 'w' for white edge color
        markerfacecolor=col, marker='.'            # Colored markers
    )
    
    # Plot the cluster center
    ax.plot(
        cluster_center[0], cluster_center[1], 'o',  # Circle marker
        markerfacecolor=col,                        # Same color as cluster
        markeredgecolor='k',                        # Black edge
        markersize=6                                # Larger size
    )

# Add title and clean up axes
ax.set_title('KMeans Clustering Results')
ax.set_xticks(())  # Remove x-axis ticks
ax.set_yticks(())  # Remove y-axis ticks

# Display the final plot
plt.show()

# ==============================================
# EXAMPLE 2: Customer Segmentation
# ==============================================

# Load customer data from online source
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv"
cust_df = pd.read_csv(url)  # Read CSV into DataFrame
print("Data loaded successfully. First 5 rows:")
print(cust_df.head())  # Show first 5 rows

# Preprocess the data by removing address column
df = cust_df.drop('Address', axis=1)  # axis=1 indicates column drop
print("\nData after dropping 'Address' column:")
print(df.head())

# Prepare data for clustering:
X = df.values[:, 1:]  # Extract all columns except Customer ID
X = np.nan_to_num(X)  # Replace any NaN values with zeros

# Standardize the features (mean=0, std=1)
Clus_dataSet = StandardScaler().fit_transform(X)
print("\nStandardized data sample:")
print(Clus_dataSet[:5])  # Show first 5 standardized samples

# Perform KMeans clustering with 3 clusters
clusterNum = 3
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(X)  # Train the model
labels = k_means.labels_  # Get cluster assignments
print("\nCluster labels for first 10 customers:")
print(labels[:10])

# Add cluster labels back to original DataFrame
df["Clus_km"] = labels
print("\nData with cluster labels:")
print(df.head(5))  # Show first 5 rows with cluster labels

# Calculate mean values for each feature by cluster
print("\nCluster means:")
print(df.groupby('Clus_km').mean())

# Create 2D visualization: Age vs Income
plt.figure(figsize=(8, 6))
area = np.pi * (X[:, 1])**2  # Use age to determine marker size
plt.scatter(
    X[:, 0], X[:, 3],        # Age (x) vs Income (y)
    s=area,                  # Marker size based on age
    c=labels.astype(float),  # Color by cluster
    alpha=0.5                # Semi-transparent markers
)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Income', fontsize=14)
plt.title('Customer Segmentation - Age vs Income', fontsize=16)
plt.tight_layout()  # Adjust layout to prevent label clipping
plt.show()

# Create 3D visualization: Education vs Age vs Income
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot
ax.scatter(
    X[:, 1], X[:, 0], X[:, 3],  # Education, Age, Income
    c=labels.astype(float),     # Color by cluster
    alpha=0.5                   # Semi-transparent markers
)
# Set axis labels
ax.set_xlabel('Education', fontsize=12)
ax.set_ylabel('Age', fontsize=12)
ax.set_zlabel('Income', fontsize=12)
ax.set_title('3D Customer Segmentation', fontsize=16)
plt.tight_layout()  # Adjust layout
plt.show()
