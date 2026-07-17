import numpy as np

from ae05_clustering import kmeans_numpy


# --------------------------------------------------
# Cluster stability via centroid consistency
# --------------------------------------------------
def stability_test(X_embed, n_clusters=5, n_runs=10, sample_frac=0.9):
    all_centroids = []

    n_samples = X_embed.shape[0]

    for _ in range(n_runs):
        idx = np.random.choice(
            n_samples,
            size=int(sample_frac * n_samples),
            replace=False
        )
        X_sub = X_embed[idx]

        labels, centroids = kmeans_numpy(
            X_sub,
            n_clusters=n_clusters
        )
        all_centroids.append(centroids)

    distances = []

    for i in range(len(all_centroids) - 1):
        c1 = all_centroids[i]
        c2 = all_centroids[i + 1]

        for j in range(c1.shape[0]):
            d = np.linalg.norm(c1[j] - c2, axis=1)
            distances.append(np.min(d))

    return np.mean(distances), np.std(distances)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from ae02_preprocessing import preprocess
    from ae04_dimensionality_reduction import reduce_dimensions

    # IMPORTANT: correct delimiter
    df = pd.read_csv("Mendeley_ae.csv", sep=";")

    X_scaled, _ = preprocess(df)
    X_2d, _, _ = reduce_dimensions(X_scaled)

    mean_disp, std_disp = stability_test(
        X_2d,
        n_clusters=5,
        n_runs=10
    )

    print(
        f"Cluster stability (centroid displacement) = "
        f"{mean_disp:.4f} ± {std_disp:.4f}"
    )
