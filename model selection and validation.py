import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path="Mendeley_ae.csv"):
    df = pd.read_csv(path, sep=None, engine='python')

    print("Loaded dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    feature_cols = ['A', 'R', 'D', 'CNTS', 'E', 'F']
    X = df[feature_cols].values.astype(float)

    print("Feature matrix shape:", X.shape)
    return X


# ============================================================
# 2. STANDARDIZATION (Manual)
# ============================================================

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std


# ============================================================
# 3. MANUAL K-MEANS
# ============================================================

def kmeans(X, k=4, max_iter=100):
    np.random.seed(42)

    n_samples = X.shape[0]
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):

        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0)
            if np.any(labels == i)
            else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# ============================================================
# 4. INERTIA (ELBOW METHOD)
# ============================================================

def compute_inertia(X, labels, centroids):
    inertia = 0
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        inertia += np.sum((cluster_points - centroids[i])**2)
    return inertia


def elbow_curve(X, k_range=range(2, 9)):
    inertias = []

    for k in k_range:
        labels, centroids = kmeans(X, k=k)
        inertias.append(compute_inertia(X, labels, centroids))

    plt.figure()
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve for Model Selection")
    plt.show()


# ============================================================
# 5. MANUAL SILHOUETTE SCORE
# ============================================================

def silhouette_score_manual(X, labels):
    n = len(X)
    unique_labels = np.unique(labels)

    silhouette_vals = []

    for i in range(n):
        same_cluster = X[labels == labels[i]]
        other_clusters = [X[labels == lbl] for lbl in unique_labels if lbl != labels[i]]

        if len(same_cluster) <= 1:
            silhouette_vals.append(0)
            continue

        a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))

        b = np.min([
            np.mean(np.linalg.norm(cluster - X[i], axis=1))
            for cluster in other_clusters
        ])

        s = (b - a) / max(a, b)
        silhouette_vals.append(s)

    return np.mean(silhouette_vals)


def silhouette_comparison_curve(X, k_range=range(2, 9)):
    scores = []

    for k in k_range:
        labels, _ = kmeans(X, k=k)
        score = silhouette_score_manual(X, labels)
        scores.append(score)

    plt.figure()
    plt.plot(list(k_range), scores, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.show()


# ============================================================
# 6. BOOTSTRAP JACCARD STABILITY
# ============================================================

def jaccard_index(a, b):
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0


def bootstrap_stability(X, k=4, n_boot=10):
    labels_full, _ = kmeans(X, k=k)

    cluster_scores = {i: [] for i in range(k)}

    for _ in range(n_boot):
        idx = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[idx]

        labels_boot, _ = kmeans(X_boot, k=k)

        for cluster_id in range(k):
            full_set = set(np.where(labels_full == cluster_id)[0])
            boot_set = set(idx[np.where(labels_boot == cluster_id)[0]])

            score = jaccard_index(full_set, boot_set)
            cluster_scores[cluster_id].append(score)

    means = []
    stds = []

    for cid in range(k):
        mean = np.mean(cluster_scores[cid])
        std = np.std(cluster_scores[cid])

        means.append(mean)
        stds.append(std)

        print(f"Cluster {cid}: {mean:.3f} ± {std:.3f}")

    # Plot
    plt.figure()
    plt.bar(range(k), means, yerr=stds, capsize=5)
    plt.ylim(0, 1)
    plt.xlabel("Cluster")
    plt.ylabel("Jaccard Stability")
    plt.title("Cluster Stability (Bootstrap Jaccard)")
    plt.show()


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    X = load_data("Mendeley_ae.csv")
    X = standardize(X)

    print("\nRunning elbow method...")
    elbow_curve(X)

    print("\nRunning silhouette comparison...")
    silhouette_comparison_curve(X)

    print("\nEvaluating k=4 stability...")
    bootstrap_stability(X, k=4)
