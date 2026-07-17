import numpy as np
import pandas as pd

# ==========================================================
# DISPLAY SETTINGS
# ==========================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# ==========================================================
# LOAD DATA
# ==========================================================

FILE_PATH = "Mendeley_ae.csv"

df = pd.read_csv(FILE_PATH, sep=None, engine='python')

print("\nDataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==========================================================
# FEATURES
# ==========================================================

feature_cols = ["A", "R", "D", "CNTS", "E", "F"]

X_original = df[feature_cols].values.astype(float)

# ==========================================================
# STANDARDIZATION
# ==========================================================

X = (X_original - np.mean(X_original, axis=0)) / np.std(X_original, axis=0)

# ==========================================================
# MANUAL K-MEANS
# ==========================================================

def kmeans(X, k=4, max_iter=100):

    np.random.seed(42)

    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):

        distances = np.linalg.norm(
            X[:, None] - centroids,
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        new_centroids = []

        for i in range(k):

            cluster_points = X[labels == i]

            if len(cluster_points) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(
                    cluster_points.mean(axis=0)
                )

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels

# ==========================================================
# CLUSTERING
# ==========================================================

labels = kmeans(X, k=4)

df_clusters = df.copy()
df_clusters["Cluster"] = labels

# ==========================================================
# CLUSTER MEANS
# ==========================================================

cluster_means = (
    df_clusters
    .groupby("Cluster")[feature_cols]
    .mean()
)

cluster_means = cluster_means.round(2)

print("\n")
print("=" * 80)
print("TABLE X. CLUSTER-WISE MEAN AE FEATURE VALUES")
print("=" * 80)
print(cluster_means)

# ==========================================================
# SAVE CSV
# ==========================================================

cluster_means.to_csv("Cluster_Feature_Table.csv")

print("\nSaved:")
print("Cluster_Feature_Table.csv")

# ==========================================================
# PERCENTAGE DIFFERENCES
# ==========================================================

print("\n")
print("=" * 80)
print("MAXIMUM FEATURE DIFFERENCES BETWEEN CLUSTERS")
print("=" * 80)

for feature in feature_cols:

    values = cluster_means[feature]

    vmax = values.max()
    vmin = values.min()

    max_cluster = values.idxmax()
    min_cluster = values.idxmin()

    pct_diff = ((vmax - vmin) / vmin) * 100

    print(
        f"{feature}: "
        f"Cluster {max_cluster} vs Cluster {min_cluster} "
        f"= {pct_diff:.1f}% difference"
    )

# ==========================================================
# IDENTIFY DOMINANT CLUSTERS
# ==========================================================

print("\n")
print("=" * 80)
print("DOMINANT FEATURE PER CLUSTER")
print("=" * 80)

for cluster in cluster_means.index:

    row = cluster_means.loc[cluster]

    dominant_feature = row.idxmax()
    dominant_value = row.max()

    print(
        f"Cluster {cluster}: "
        f"Highest feature = {dominant_feature} "
        f"({dominant_value:.2f})"
    )

print("\nAnalysis Complete.")