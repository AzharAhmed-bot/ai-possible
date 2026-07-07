import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

os.makedirs("figures", exist_ok=True)

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv('Data_Visualization.csv')
print(f"Dataset shape: {df.shape}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
features = df.select_dtypes(include='number')
features = features.dropna(axis=1, how='all')       

imputer = SimpleImputer(strategy='mean')             
imputed = imputer.fit_transform(features)

scaler = StandardScaler()
scaled = scaler.fit_transform(imputed)

# ── 3. Clustering (K-Means, k=3) ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled)

# ── 4. Feature distributions — histograms ────────────────────────────────────
key_props = ['Al', 'Si', 'Cu', 'Mg', 'Ni', 'Mn',
             'Vf_FCC_A1', 'eut. frac.[%]', 'T(liqu)', 'T(sol)', 'delta_T']
key_props = [c for c in key_props if c in features.columns]

features[key_props].hist(figsize=(16, 10), bins=40, color='steelblue', edgecolor='white')
plt.suptitle("Distribution of Key Material Properties", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("figures/01_histograms.png", bbox_inches='tight')
plt.close()

# ── 5. Correlation heatmap ────────────────────────────────────────────────────
plt.figure(figsize=(14, 11))
sns.heatmap(features[key_props].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig("figures/02_correlation_heatmap.png")
plt.close()

# ── 6. PCA — plain projection ─────────────────────────────────────────────────
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
var1, var2 = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3, s=3, color='steelblue')
plt.xlabel(f"PC1 ({var1:.1f}% variance)")
plt.ylabel(f"PC2 ({var2:.1f}% variance)")
plt.title("PCA — All Simulated Candidate Materials")
plt.tight_layout()
plt.savefig("figures/03_pca_overview.png")
plt.close()

# ── 7. PCA coloured by cluster ────────────────────────────────────────────────
cluster_names = {0: "High-Cu (Aerospace)", 1: "Lean/Balanced", 2: "High-Si (Casting)"}
colors = ['tab:blue', 'tab:orange', 'tab:green']

plt.figure(figsize=(9, 7))
for i in range(3):
    mask = labels == i
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1],
                label=f"Cluster {i}: {cluster_names[i]}",
                alpha=0.3, s=3, color=colors[i])
plt.xlabel(f"PC1 ({var1:.1f}% variance)")
plt.ylabel(f"PC2 ({var2:.1f}% variance)")
plt.title("PCA — K-Means Clusters (k=3)")
plt.legend(markerscale=4, loc='upper left')
plt.tight_layout()
plt.savefig("figures/04_pca_clusters.png")
plt.close()

# ── 8. Scree plot — how much variance each PC captures ───────────────────────
pca_full = PCA().fit(scaled)
cumvar = pca_full.explained_variance_ratio_.cumsum() * 100

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o', markersize=3)
plt.axhline(95, color='red', linestyle='--', label='95% threshold')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Scree Plot — PCA Explained Variance")
plt.legend()
plt.tight_layout()
plt.savefig("figures/05_scree_plot.png")
plt.close()

# ── 9. Parallel coordinates — multi-dimensional overview ─────────────────────
from pandas.plotting import parallel_coordinates

sample = features[key_props].copy()
sample['Cluster'] = [cluster_names[l] for l in labels]
sample_small = sample.groupby('Cluster', group_keys=False).apply(
    lambda g: g.sample(min(500, len(g)), random_state=42)
)

plt.figure(figsize=(16, 6))
parallel_coordinates(sample_small, class_column='Cluster',
                     color=colors, alpha=0.05, linewidth=0.8)
plt.xticks(rotation=30, ha='right')
plt.title("Parallel Coordinates — Material Properties by Cluster")
plt.tight_layout()
plt.savefig("figures/06_parallel_coordinates.png")
plt.close()

# ── 10. Cluster summary — radar / bar chart ───────────────────────────────────
df_clustered = features[key_props].copy()
df_clustered['cluster'] = labels
cluster_means = df_clustered.groupby('cluster').mean()
cluster_means.index = [cluster_names[i] for i in cluster_means.index]

cluster_means.T.plot(kind='bar', figsize=(14, 6), width=0.7, colormap='Set2')
plt.title("Average Property Values per Cluster")
plt.ylabel("Mean Value")
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("figures/07_cluster_means.png")
plt.close()

print("\nAll figures saved to figures/")
print("\nCluster sizes:")
import numpy as np
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u} ({cluster_names[u]}): {c:,} materials")
print(f"\nPCA: PC1={var1:.1f}%, PC2={var2:.1f}% — total={var1+var2:.1f}% variance captured")
