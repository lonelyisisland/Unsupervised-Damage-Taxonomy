import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD DATASET
# ==========================================================

def load_dataset(filepath):
    # Try default separator
    df = pd.read_csv(filepath)

    # If single column detected, likely semicolon separated
    if df.shape[1] == 1:
        df = pd.read_csv(filepath, sep=';')

    print("Loaded dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df


# ==========================================================
# 2. PREPROCESS FEATURES
# ==========================================================

def preprocess_features(df):

    feature_cols = ["A", "R", "D", "CNTS", "E", "F"]

    X = df[feature_cols].values.astype(float)

    # Standardization
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    return X


# ==========================================================
# 3. MANUAL K-MEANS
# ==========================================================

def kmeans(X, k=4, max_iter=100):
    n = X.shape[0]
    centroids = X[np.random.choice(n, k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i)
            else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# ==========================================================
# 4. SILHOUETTE SCORE (MANUAL)
# ==========================================================

def silhouette_score_manual(X, labels):

    n = len(X)
    unique_clusters = np.unique(labels)
    silhouette_vals = np.zeros(n)

    for i in range(n):

        same_cluster = labels == labels[i]

        # a(i): intra-cluster distance
        if np.sum(same_cluster) > 1:
            a = np.mean(np.linalg.norm(X[i] - X[same_cluster], axis=1))
        else:
            a = 0

        # b(i): nearest-cluster distance
        b = np.inf
        for c in unique_clusters:
            if c != labels[i]:
                cluster_mask = labels == c
                if np.any(cluster_mask):
                    dist = np.mean(np.linalg.norm(X[i] - X[cluster_mask], axis=1))
                    b = min(b, dist)

        silhouette_vals[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

    return silhouette_vals.mean(), silhouette_vals


# ==========================================================
# 5. MAIN
# ==========================================================

if __name__ == "__main__":

    filepath = "Mendeley_ae.csv"  # Make sure file is in same folder

    df = load_dataset(filepath)

    X = preprocess_features(df)

    print("Feature matrix shape:", X.shape)

    # Run k-means with k=4
    k = 4
    labels, centroids = kmeans(X, k=k)

    # Compute silhouette
    sil_score, sil_values = silhouette_score_manual(X, labels)

    print(f"\nSilhouette Score (k={k}): {sil_score:.3f}")

    # ======================================================
    # 6. SILHOUETTE PLOT
    # ======================================================

    plt.figure(figsize=(6,5))

    y_lower = 10

    for i in range(k):
        cluster_sil = sil_values[labels == i]
        cluster_sil.sort()

        size_cluster = len(cluster_sil)
        y_upper = y_lower + size_cluster

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
        y_lower = y_upper + 10

    plt.axvline(sil_score, linestyle='--')
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster label")
    plt.title(f"Silhouette Plot (k={k})")
    plt.tight_layout()
    plt.show()
