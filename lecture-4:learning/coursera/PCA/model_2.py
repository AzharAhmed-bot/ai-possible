from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np



iris=load_iris()
evidence=iris.data
target=iris.target
target_names=iris.target_names


scaler=StandardScaler()
std_evidence=scaler.fit_transform(evidence)
model=PCA(n_components=2)
x_pca=model.fit_transform(std_evidence)
print(x_pca)
print(x_pca[0,0])
print(x_pca[1,0])
print(x_pca[2,0])


plt.figure()
colors=['navy','turquoise','darkorange']
lw=1

for color,i, target_name in zip(colors,[0,1,2],target_names):
    plt.scatter(x_pca[target==i,0],x_pca[target==i,1],color=color,s=50,ec='k',alpha=0.7,lw=lw,label=target_name)

plt.title("PCA 2-Dimensional reduction of IRIS dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.savefig("iris.png")

#percentage of the original feature space variance done by the two combined principal components
print(100 * model.explained_variance_ratio_.sum())


explained_variance_ratio=model.explained_variance_ratio_
print(explained_variance_ratio)
print(len(explained_variance_ratio))
plt.figure(figsize=(10,6))
plt.bar(x=range(1,len(explained_variance_ratio)+1),height=explained_variance_ratio,alpha=1,align='center',label='PCA explained variance ratio')
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.title("Explained Variance Ratio of IRIS dataset")
cumulative_variance=np.cumsum(explained_variance_ratio)
plt.step(range(1,len(cumulative_variance)+1), cumulative_variance, where='mid', linestyle='--', lw=3,color='red', label='Cumulative Explained Variance')
plt.xticks(range(1,len(cumulative_variance)+1))
plt.legend()
plt.savefig("explained_variance_ratio.png")