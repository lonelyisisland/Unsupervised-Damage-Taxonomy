import numpy as np


# ----------------------------------------
# NumPy-only K-Means implementation
# ----------------------------------------
def kmeans_numpy(X, n_clusters=5, max_iter=100, random_state=42):
    np.random.seed(random_state)

    # Randomly initialize centroids
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        # Assign clusters
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
            for k in range(n_clusters)
        ])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# ----------------------------------------
# Wrapper function (same interface idea)
# ----------------------------------------
def cluster_events(X_2d, n_clusters=5):
    labels, centroids = kmeans_numpy(
        X_2d,
        n_clusters=n_clusters
    )
    return labels, centroids


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from ae02_preprocessing import preprocess
    from ae04_dimensionality_reduction import reduce_dimensions

    # IMPORTANT: semicolon separator
    df = pd.read_csv("Mendeley_ae.csv", sep=";")

    X, _ = preprocess(df)
    X_2d, _, _ = reduce_dimensions(X)

    labels, centroids = cluster_events(X_2d, n_clusters=5)

    print("Clusters found:", len(np.unique(labels)))


