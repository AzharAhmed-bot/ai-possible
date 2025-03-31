import  numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Patch
from matplotlib import cm

def evaluate_clustering(X, labels, n_clusters, ax=None, title_suffix=''):
    """
    Evaluate a clustering model using silhouette scores and the Davies-Bouldin index.
    
    Parameters:
    X (ndarray): Feature matrix.
    labels (array-like): Cluster labels assigned to each sample.
    n_clusters (int): The number of clusters in the model.
    ax: The subplot axes to plot on.
    title_suffix (str): Optional suffix for plot titlec
    
    Returns:
    None: Displays silhoutte scores and a silhouette plot.
    """
    if ax is None:
        ax = plt.gca()  # Get the current axis if none is provided
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    # Plot silhouette analysis on the provided axis
    unique_labels = np.unique(labels)
    colormap = cm.tab10
    color_dict = {label: colormap(float(label) / n_clusters) for label in unique_labels}
    y_lower = 10
    for i in unique_labels:
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = color_dict[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Score for {title_suffix} \n' + 
                 f'Average Silhouette: {silhouette_avg:.2f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlim([-0.25, 1])  # Set the x-axis range to [0, 1]

    ax.set_yticks([])


evidence,label=make_blobs(n_samples=500,n_features=3,centers=4,cluster_std=[1.0,3,5,2],random_state=42)
kmeans=KMeans(n_clusters=4,random_state=42)
predictions=kmeans.fit_predict(evidence)
colormap=cm.tab10



# Plot the blobs
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.scatter(evidence[:,0],evidence[:,1],c='k',ec='k',alpha=0.7)
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='r',ec='k',marker='X',s=200,label="Centroids",alpha=0.9)
plt.title(f"Synthetic data with 4 clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.savefig("blobs.png")


plt.scatter(
    centers[:,0],
    centers[:,1],
    marker='o',
    c='white',
    alpha=1,
    s=200,
    label="centroids"
)
# Label the custer number
for i, c in enumerate(centers):
    plt.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

plt.title(f'KMeans Clustering with 4 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Evaluate the clustering
plt.subplot(1, 3, 3)
evaluate_clustering(evidence, predictions, 4, title_suffix=' k-Means Clustering')
plt.savefig('evaluation.png')



# Inertia --- How stable is our model
n_runs=8
inertia_values=[]
inertia_values_2=[]
silhouette_scores=[]
davies_bouldin_indices=[]

# Plot the inertia values
n_cols=2
n_rows=-(-n_runs // n_cols)
plt.figure(figsize=(16, 16))
for i in range(n_runs):
    kmeans_2=KMeans(n_clusters=4,random_state=None)
    kmeans_2.fit(evidence)
    inertia_values.append(kmeans_2.inertia_)
    plt.subplot(n_rows, n_cols, i+1)
    plt.scatter(evidence[:,0],evidence[:,1],c=kmeans.labels_,cmap='tab10',alpha=0.6,ec='k')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='r',ec='k',marker='X',s=200,label="Centroids",alpha=0.9)
    plt.title(f"K-Means Run {i+1}")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper right',fontsize='small')
    plt.tight_layout()

plt.savefig('inertia.png')



#
k_values=range(2,11)
for k in k_values:
    kmeans_3=KMeans(n_clusters=k,random_state=None)
    y_pred=kmeans_3.fit_predict(evidence)
    inertia_values_2.append(kmeans_3.inertia_)
    silhouette_scores.append(silhouette_score(evidence,y_pred))
    davies_bouldin_indices.append(davies_bouldin_score(evidence,y_pred))

plt.figure(figsize=(18,16))
plt.subplot(1,3,1)
plt.plot(k_values,inertia_values_2,marker='o')
plt.title('Elbow Method: Inertia vs. k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1,3,2)
plt.plot(k_values,silhouette_scores,marker='o')
plt.title('Silhouette Coefficient vs. k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Coefficient')

plt.subplot(1,3,3)
plt.plot(k_values,davies_bouldin_indices,marker='o')
plt.title('Davies-Bouldin Index vs. k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.savefig('elbow.png')










