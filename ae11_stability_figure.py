import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD DATA
# ==========================================================
def load_data(filepath):
    df = pd.read_csv(filepath, sep=None, engine="python")

    if "Damage Mechanism" in df.columns:
        df = df.drop(columns=["Damage Mechanism"])

    df = df.dropna()
    return df.values


# ==========================================================
# 2. SIMPLE PCA (NUMPY ONLY)
# ==========================================================
def pca_numpy(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    return np.dot(X_centered, eigvecs[:, :n_components])


# ==========================================================
# 3. SIMPLE KMEANS (NO SKLEARN)
# ==========================================================
def kmeans_numpy(X, k=5, n_iter=100):
    np.random.seed()
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(n_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i)
            else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


# ==========================================================
# 4. JACCARD SIMILARITY
# ==========================================================
def jaccard_similarity(setA, setB):
    intersection = len(setA.intersection(setB))
    union = len(setA.union(setB))
    if union == 0:
        return 0
    return intersection / union


# ==========================================================
# 5. BOOTSTRAP STABILITY (JACCARD BASED)
# ==========================================================
def bootstrap_jaccard(X_embed, k=5, runs=50):

    base_labels = kmeans_numpy(X_embed, k=k)

    base_clusters = {
        cid: set(np.where(base_labels == cid)[0])
        for cid in np.unique(base_labels)
    }

    cluster_scores = {cid: [] for cid in base_clusters.keys()}

    for _ in range(runs):

        # Bootstrap sample indices
        idx = np.random.choice(len(X_embed), len(X_embed), replace=True)
        X_sample = X_embed[idx]

        sample_labels = kmeans_numpy(X_sample, k=k)

        sample_clusters = {
            cid: set(idx[np.where(sample_labels == cid)[0]])
            for cid in np.unique(sample_labels)
        }

        # Match clusters by maximum overlap
        for cid in base_clusters.keys():

            best_score = 0

            for sid in sample_clusters.keys():
                score = jaccard_similarity(
                    base_clusters[cid],
                    sample_clusters[sid]
                )
                if score > best_score:
                    best_score = score

            cluster_scores[cid].append(best_score)

    return cluster_scores


# ==========================================================
# 6. PLOT FIGURE 8
# ==========================================================
def plot_jaccard(cluster_scores):

    means = []
    stds = []
    labels = []

    for cid, scores in cluster_scores.items():
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        labels.append(f"Cluster {cid}")

    x = np.arange(len(labels))

    plt.figure(figsize=(7,4))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Jaccard similarity")
    plt.ylim(0,1)
    plt.title("Cluster Stability Analysis (Bootstrap Jaccard)")
    plt.tight_layout()
    plt.show()

    for i, cid in enumerate(cluster_scores.keys()):
        print(f"Cluster {cid} → Mean Jaccard: {means[i]:.3f} ± {stds[i]:.3f}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    X = load_data("Mendeley_ae.csv")
    X_embed = pca_numpy(X, n_components=2)

    cluster_scores = bootstrap_jaccard(X_embed, k=5, runs=10)

    plot_jaccard(cluster_scores)
