import numpy as np
import pandas as pd


# --------------------------------------------------
# NumPy-only hierarchical grouping of clusters
# --------------------------------------------------
def build_hierarchy(feature_df, cluster_labels, n_phases=4):
    """
    Build a higher-level damage hierarchy by clustering
    cluster centroids using NumPy-only logic.
    """

    df = feature_df.copy()
    df["cluster"] = cluster_labels

    # Remove noise if any (not expected with K-means)
    df = df[df["cluster"] != -1]

    # Compute cluster centroids (physical feature space)
    cluster_centroids = df.groupby("cluster").mean().values
    cluster_ids = df.groupby("cluster").mean().index.values

    # Pairwise Euclidean distance matrix
    n = cluster_centroids.shape[0]
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(
                cluster_centroids[i] - cluster_centroids[j]
            )

    # Simple agglomerative grouping (greedy)
    phase_labels = np.zeros(n, dtype=int)

    # Sort clusters by average distance (proxy for Ward structure)
    avg_dist = dist_matrix.mean(axis=1)
    sorted_idx = np.argsort(avg_dist)

    # Assign phases sequentially
    splits = np.array_split(sorted_idx, n_phases)

    for phase_id, indices in enumerate(splits):
        for idx in indices:
            phase_labels[idx] = phase_id + 1  # phases start from 1

    # Map cluster → phase
    phase_map = dict(zip(cluster_ids, phase_labels))

    return phase_map


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    from ae02_preprocessing import preprocess
    from ae04_dimensionality_reduction import reduce_dimensions
    from ae05_clustering import cluster_events

    # IMPORTANT: correct delimiter
    df = pd.read_csv("Mendeley_ae.csv", sep=";")

    # Preprocess
    X_scaled, _ = preprocess(df)

    # Physical feature dataframe (for hierarchy)
    feature_df = df[["A", "R", "D", "CNTS", "E", "F"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # Dimensionality reduction + clustering
    X_2d, _, _ = reduce_dimensions(X_scaled)
    labels, _ = cluster_events(X_2d, n_clusters=5)

    # Build hierarchy
    phase_map = build_hierarchy(
        feature_df,
        labels,
        n_phases=4
    )

    print("\nCluster → Phase mapping")
    for k, v in phase_map.items():
        print(f"  Cluster {k} → Phase {v}")
