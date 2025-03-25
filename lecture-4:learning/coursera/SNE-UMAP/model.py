import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

centers = [ [ 2, -6, -6],
            [-1,  9,  4],
            [-8,  7,  2],
            [ 4,  7,  9] ]

cluster_std=[1,1,2,3.5]

X,labels_=make_blobs(n_samples=500,centers=centers,n_features=3,cluster_std=cluster_std,random_state=42)

scaler=StandardScaler()
x_scaled=scaler.fit_transform(X)



#T-SNE
# n_components: The number of dimensions for the output (default = 2).
# perplexity: Balances attention between local and global aspects of the data (default = 30).
# learning_rate: Controls the step size during optimization (default = 200).
tsne_model=TSNE(n_components=2,random_state=42,perplexity=30,max_iter=1000)
x_tsne=tsne_model.fit_transform(x_scaled)
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.scatter(x_tsne[:,0],x_tsne[:,1],c=labels_,cmap="viridis",s=50,alpha=0.7,ec='k')
ax.set_title("2D t-SNE Projection of 3D Data")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig('tsne.png')


# UMAP
# n_neighbors: Controls the local neighborhood size (default = 15).
# min_dist: Controls the minimum distance between points in the embedded space (default = 0.1).
# n_components: The dimensionality of the embedding (default = 2).
umap_model=UMAP(n_components=2,random_state=42,min_dist=0.5,spread=1,n_jobs=1)
x_umap=umap_model.fit_transform(x_scaled)
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.scatter(x_umap[:,0],x_umap[:,1],c=labels_,cmap='viridis',s=50,alpha=0.7,ec='k')
ax.set_title("2D UMAP Projection of 3D Data")
ax.set_xlabel("UMAP Component 1")
ax.set_ylabel("UMAP Component 2")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("umap.png")


# PCA
pca_model=PCA(n_components=2)
x_pca=pca_model.fit_transform(x_scaled)
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111)
ax.scatter(x_pca[:,0],x_pca[:,1],c=labels_,cmap='viridis',s=50,alpha=0.7,ec='k')
ax.set_title("2D PCA Projection of 3D Data")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_xticks([])
ax.set_yticks([])
plt.savefig("pca.png")

