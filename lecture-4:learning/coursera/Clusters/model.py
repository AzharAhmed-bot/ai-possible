import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
data = data.drop('Address', axis=1).dropna()

# Standardize the features
evidence = data.values[:, 1:]
std_scaler = StandardScaler()
std_data = std_scaler.fit_transform(evidence)

# K-Means clustering
model = KMeans(init='k-means++', n_clusters=3, n_init=12, random_state=42)
model.fit(std_data)

labels = model.labels_
cluster_centers = model.cluster_centers_

# Plot
fig, ax = plt.subplots(figsize=(6, 4))

colors = plt.cm.tab10(np.linspace(0, 1, model.n_clusters))

for k, col in zip(range(model.n_clusters), colors):
    my_members = (labels == k)
    cluster_center = cluster_centers[k]

    # Plot data points
    ax.scatter(std_data[my_members, 0], std_data[my_members, 1], c=[col], label=f'Cluster {k}', edgecolors='k', alpha=0.6)

    # Plot centroids
    ax.scatter(cluster_center[0], cluster_center[1], c=[col], edgecolors='black', marker='X', s=200, label=f'Centroid {k}')

ax.set_title('K-Means Clustering')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
plt.savefig('kmeans.png')
