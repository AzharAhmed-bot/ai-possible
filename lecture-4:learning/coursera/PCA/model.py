import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



np.random.seed(42)
mean=[0,0]
cov=[[3,2],[2,2]]
X=np.random.multivariate_normal(mean=mean,cov=cov,size=100)
print("X values:")
print(X)

plt.figure()
plt.scatter(X[:,0],X[:,1],edgecolors='k',alpha=0.7)
plt.title("Scatter Plot of Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.grid(True)
plt.savefig("bivariate.png") 

pca=PCA(n_components=2)
x_pca=pca.fit_transform(X)

components=pca.components_
print("Components 1:")
print(components[0])
print("Components 2:")
print(components[1])

# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)

projection_pc1=np.dot(X,components[0])
projection_pc2=np.dot(X,components[1])
# The projections explain how much the data point is aligned with the Principle component
# It explain it in terms of magnitude and direction
# In this case data point 1 and 2 are strongly aligned with PC1.
# This means it captures the most significant variance direction in the dataset
# Thus is shows how much the data points aligns with principle component
print("Projection on PC1 and PC2:")
print(projection_pc1)
print(projection_pc2)

# this step takes the projection values 
# and reconstructs where they would fall in the original 2D space 
# based on the direction of PC1.
x_pc1=projection_pc1 * components[0][0]
y_pc1=projection_pc1 * components[0][1]
x_pc2=projection_pc2 * components[1][0]
y_pc2=projection_pc2 * components[1][1]



# Plot the original data
plt.figure()
plt.scatter(X[:,0],X[:,1],label="Original Data",ec='k',alpha=0.7)


# Plot the projection along PC1 and PC2
plt.scatter(x_pc1,y_pc1,c='r',ec='k',marker='X',s=70,alpha=0.5,label="Projection on PC1")
plt.scatter(x_pc2,y_pc2,c='b',ec='k',marker='X',s=70,alpha=0.5,label="Projection on PC2")
plt.title('Linearly Correlated Data Projected onto Principal Components', )
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.savefig("projection.png")
