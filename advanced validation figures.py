import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# ==================================================
# FIGURE A — CLUSTER STABILITY (CENTROID DISPLACEMENT)
# ==================================================
def cluster_centroid_stability(X_2d, cluster_fn, n_runs=10, sample_frac=0.8):
    """
    Stability based on centroid displacement (NumPy-only)
    """
    centroids_all = []

    for seed in range(n_runs):
        np.random.seed(seed)
        idx = np.random.choice(
            len(X_2d),
            int(sample_frac * len(X_2d)),
            replace=False
        )
        labels, _ = cluster_fn(X_2d[idx])

        centroids = []
        for c in np.unique(labels):
            centroids.append(X_2d[idx][labels == c].mean(axis=0))

        centroids_all.append(np.array(centroids))

    # Align centroid counts (min clusters)
    min_k = min(c.shape[0] for c in centroids_all)
    centroids_all = [c[:min_k] for c in centroids_all]

    displacements = []
    for i in range(len(centroids_all) - 1):
        disp = np.linalg.norm(
            centroids_all[i] - centroids_all[i + 1],
            axis=1
        ).mean()
        displacements.append(disp)

    return np.mean(displacements), np.std(displacements)


# ==================================================
# FIGURE B — FEATURE DISTRIBUTIONS (BOXPLOTS)
# ==================================================
def plot_feature_distributions(feature_df, labels):
    df = feature_df.copy()
    df["cluster"] = labels

    for col in feature_df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="cluster", y=col, data=df)
        plt.title(f"Distribution of AE feature '{col}' across clusters")
        plt.tight_layout()
        plt.show()


# ==================================================
# FIGURE C — CUMULATIVE DAMAGE INDICATOR
# ==================================================
def plot_cumulative_damage(feature_df, feature="E"):
    cum_feature = feature_df[feature].cumsum()

    plt.figure(figsize=(8, 4))
    plt.plot(cum_feature, linewidth=2)
    plt.xlabel("Event index (fatigue progression)")
    plt.ylabel(f"Cumulative {feature}")
    plt.title(f"Cumulative damage indicator based on '{feature}'")
    plt.tight_layout()
    plt.show()


# ==================================================
# FIGURE D — TRANSITION PROBABILITIES (MARKOV-READY)
# ==================================================
def extract_transitions(labels):
    transitions = []
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            transitions.append((int(labels[i]), int(labels[i + 1])))
    return Counter(transitions)


def plot_transition_probabilities(transitions, n_clusters):
    mat = np.zeros((n_clusters, n_clusters))

    for (i, j), c in transitions.items():
        mat[i, j] += c

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat = mat / row_sums

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label="Transition probability")
    plt.xlabel("Next damage state")
    plt.ylabel("Current damage state")
    plt.title("Markov transition probability matrix")
    plt.tight_layout()
    plt.show()


# ==================================================
# MAIN — RUN EVERYTHING
# ==================================================
if __name__ == "__main__":

    from ae02_preprocessing import preprocess
    from ae04_dimensionality_reduction import reduce_dimensions
    from ae05_clustering import cluster_events

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    df = pd.read_csv("Mendeley_ae.csv", sep=";")

    # ------------------------------
    # PREPROCESS
    # ------------------------------
    X_scaled, _ = preprocess(df)

    # ------------------------------
    # DIMENSIONALITY REDUCTION
    # ------------------------------
    X_2d, _, _ = reduce_dimensions(X_scaled)

    # ------------------------------
    # CLUSTERING
    # ------------------------------
    labels, _ = cluster_events(X_2d, n_clusters=5)

    # ------------------------------
    # FEATURE DATAFRAME
    # ------------------------------
    feature_df = df[["A", "R", "D", "CNTS", "E", "F"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # ==================================================
    # FIGURE A — STABILITY
    # ==================================================
    mean_disp, std_disp = cluster_centroid_stability(
        X_2d,
        lambda X: cluster_events(X, n_clusters=5),
        n_runs=10
    )

    print(f"Cluster centroid displacement: {mean_disp:.4f} ± {std_disp:.4f}")

    # ==================================================
    # FIGURE B — FEATURE DISTRIBUTIONS
    # ==================================================
    plot_feature_distributions(feature_df, labels)

    # ==================================================
    # FIGURE C — CUMULATIVE DAMAGE
    # ==================================================
    plot_cumulative_damage(feature_df, feature="E")

    # ==================================================
    # FIGURE D — TRANSITION PROBABILITIES
    # ==================================================
    transitions = extract_transitions(labels)
    plot_transition_probabilities(
        transitions,
        n_clusters=len(np.unique(labels))
    )
