import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD DATA
# ==========================================================

df = pd.read_csv("Mendeley_ae.csv", sep=';')
features = ["A", "R", "D", "CNTS", "E", "F"]
X = df[features].values.astype(float)

# ==========================================================
# 2. STANDARDIZATION
# ==========================================================

X = (X - X.mean(axis=0)) / X.std(axis=0)

# ==========================================================
# 3. MANUAL K-MEANS
# ==========================================================

def kmeans(X, k=4, max_iter=100):
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


# ==========================================================
# 4. BASE CLUSTERING
# ==========================================================

k = 4
base_labels, base_centroids = kmeans(X, k=k)

# ==========================================================
# 5. CENTROID MATCHING FUNCTION
# ==========================================================

def match_clusters(base_centroids, boot_centroids):
    """
    Matches bootstrap clusters to base clusters
    using minimum centroid distance.
    """
    mapping = {}
    used = set()

    for i in range(len(base_centroids)):
        distances = np.linalg.norm(boot_centroids - base_centroids[i], axis=1)
        sorted_idx = np.argsort(distances)

        for idx in sorted_idx:
            if idx not in used:
                mapping[idx] = i
                used.add(idx)
                break

    return mapping


# ==========================================================
# 6. JACCARD FUNCTION
# ==========================================================

def jaccard(set1, set2):
    if len(set1 | set2) == 0:
        return 0
    return len(set1 & set2) / len(set1 | set2)


# ==========================================================
# 7. BOOTSTRAP STABILITY
# ==========================================================

def bootstrap_stability(X, base_labels, base_centroids, n_runs=20):

    k = len(base_centroids)
    stability = {i: [] for i in range(k)}

    for run in range(n_runs):

        # Resample indices
        indices = np.random.choice(len(X), len(X), replace=True)
        X_sample = X[indices]

        boot_labels, boot_centroids = kmeans(X_sample, k=k)

        # Match clusters
        mapping = match_clusters(base_centroids, boot_centroids)

        # Relabel bootstrap clusters
        relabeled = np.zeros_like(boot_labels)

        for boot_cluster, base_cluster in mapping.items():
            relabeled[boot_labels == boot_cluster] = base_cluster

        # Compute Jaccard
        for cluster in range(k):
            base_set = set(np.where(base_labels == cluster)[0])
            sample_set = set(indices[np.where(relabeled == cluster)])

            stability[cluster].append(jaccard(base_set, sample_set))

    return stability


stability = bootstrap_stability(X, base_labels, base_centroids)

# ==========================================================
# 8. PRINT RESULTS
# ==========================================================

print("\nCluster Stability (Bootstrap Jaccard):")

means = []
stds = []

for cluster in stability:
    mean_j = np.mean(stability[cluster])
    std_j = np.std(stability[cluster])
    means.append(mean_j)
    stds.append(std_j)

    print(f"Cluster {cluster}: {mean_j:.3f} ± {std_j:.3f}")

# ==========================================================
# 9. PLOT STABILITY FIGURE (FINAL FIGURE 8)
# ==========================================================

plt.figure(figsize=(6,4))
plt.bar(range(k), means, yerr=stds, capsize=5)
plt.xticks(range(k), [f"Cluster {i}" for i in range(k)])
plt.ylabel("Jaccard similarity")
plt.title("Cluster Stability Analysis (Bootstrap)")
plt.ylim(0,1)
plt.tight_layout()
plt.show()
