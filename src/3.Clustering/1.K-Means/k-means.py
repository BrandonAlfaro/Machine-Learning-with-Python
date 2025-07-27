# ------------------------------------------------------------------------------  
# File: s_linear_regressiok_nearest_neigbors.py  
# Description: Implementation of a basic linear regression model in Python.  
# (Note: The filename and description mention linear regression and k-NN,  
# but this script actually implements KMeans clustering.)  
#  
# Author: BrandonAlfaro  
# Date: 27/1/2025  
#  
# License: MIT  
# ------------------------------------------------------------------------------

# Importing essential libraries
import random                       # Built-in Python module for random number generation
import numpy as np                  # NumPy: library for numerical computations, especially with arrays
import matplotlib.pyplot as plt     # Matplotlib: used for creating static visualizations/plots
from sklearn.cluster import KMeans  # Import KMeans clustering algorithm from scikit-learn
from sklearn.datasets import make_blobs  # Utility to generate sample data (blobs/clusters)

# Disable warnings for cleaner output
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn  # Overwrite default warning function to do nothing

# EXAMPLE 1

# Set a random seed to ensure reproducibility of the results
np.random.seed(0)

# Generate synthetic data for clustering:
# 5000 data points centered around four predefined coordinates, with a standard deviation of 0.9
X, y = make_blobs(
    n_samples=5000,                      # Number of data points
    centers=[[4,4], [-2, -1], [2, -3], [1, 1]],  # Coordinates of the true cluster centers
    cluster_std=1                        # Controls the spread of each cluster
)

# Visualize the generated data using a scatter plot
plt.scatter(X[:, 0], X[:, 1], marker='.')  # Plot x and y positions of all points
plt.show()

# Create a KMeans clustering model
k_means = KMeans(
    init="k-means++",   # Smart centroid initialization to speed up convergence
    n_clusters=4,       # We want to group the data into 4 clusters
    n_init=12           # Run the algorithm 12 times with different centroid seeds and pick the best result
)

# Fit the KMeans model to the data
k_means.fit(X)

# Get the cluster labels for each data point
k_means_labels = k_means.labels_

# Get the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_

# Create a new plot for visualizing the clustering results
fig = plt.figure(figsize=(6, 4))  # Set the size of the figure (6 inches by 4 inches)

# Define a color map to assign different colors to different clusters
# linspace creates evenly spaced values between 0 and 1 for mapping to colors
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Add a subplot (only one in this case)
ax = fig.add_subplot(1, 1, 1)

# Plot each cluster separately with its color
# Loop through each cluster index and its assigned color
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Boolean mask: True for points that belong to the k-th cluster
    my_members = (k_means_labels == k)
    
    # Get the coordinates of the k-th cluster center
    cluster_center = k_means_cluster_centers[k]
    
    # Plot all the points in the current cluster using the assigned color
    ax.plot(
        X[my_members, 0], X[my_members, 1], 'w', 
        markerfacecolor=col, marker='.'
    )
    
    # Plot the cluster center with a black edge to highlight it
    ax.plot(
        cluster_center[0], cluster_center[1], 'o', 
        markerfacecolor=col, markeredgecolor='k', markersize=6
    )

# Add title to the plot
ax.set_title('KMeans')

# Remove tick marks from both axes for a cleaner look
ax.set_xticks(())
ax.set_yticks(())

# Display the final plot
plt.show()
