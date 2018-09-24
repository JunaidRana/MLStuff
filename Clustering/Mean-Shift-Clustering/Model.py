# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:09:57 2018

@author: Junaid.raza
"""
"""
Meanshift is a clustering algorithm that assigns the datapoints to the clusters iteratively
by shifting points towards the mode. The mode can be understood as the highest density
 of datapoints (in the region, in the context of the Meanshift).



Given a set of datapoints, the algorithm iteratively assign each datapoint 
towards the closest cluster centroid. The direction to the closest cluster
 centroid is determined by where most of the points nearby are at. So each 
 iteration each data point will move closer to where the most points are at,
 which is or will lead to the cluster center. When the algorithm stops, each 
 point is assigned to a cluster.
"""
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()