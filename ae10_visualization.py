import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# FIGURE 1 — PCA EMBEDDING WITH CLUSTERS
# ==================================================
def plot_pca_embedding(X_2d, labels):
    plt.figure(figsize=(6, 5))

    for cid in np.unique(labels):
        mask = labels == cid
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=6,
            alpha=0.7,
            label=f"Cluster {cid}"
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA embedding of AE events with unsupervised clusters")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.show()


# ==================================================
# FIGURE 2 — CLUSTER-WISE FEATURE BAR PROFILES
# ==================================================
def plot_cluster_feature_profiles(feature_df, labels):
    df = feature_df.copy()
    df["cluster"] = labels
    means = df.groupby("cluster").mean()

    plt.figure(figsize=(10, 4))
    means.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean feature value")
    plt.title("Cluster-wise AE feature profiles")
    plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


# ==================================================
# FIGURE 3 — NORMALIZED RADAR (DAMAGE SIGNATURES)
# ==================================================
def plot_radar_signatures(feature_df, labels):
    features = feature_df.columns.tolist()

    means = feature_df.assign(cluster=labels).groupby("cluster").mean()
    means_norm = (means - means.min()) / (means.max() - means.min())

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    angles = np.append(angles, angles[0])

    plt.figure(figsize=(6, 6))

    for cid in means_norm.index:
        values = means_norm.loc[cid].values
        values = np.append(values, values[0])

        plt.plot(angles, values, label=f"Cluster {cid}")
        plt.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], features)
    plt.title("Normalized AE feature signatures of clusters")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


# ==================================================
# FIGURE 4 — TRANSITION MATRIX (DAMAGE EVOLUTION)
# ==================================================
def plot_transition_matrix(transitions, n_clusters):
    mat = np.zeros((n_clusters, n_clusters))

    for (i, j), count in transitions.items():
        mat[int(i), int(j)] = count

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar(label="Transition count")
    plt.xlabel("Next cluster")
    plt.ylabel("Current cluster")
    plt.title("AE cluster transition matrix")
    plt.tight_layout()
    plt.show()


# ==================================================
# FIGURE 5 — CLUSTER → PHASE HIERARCHY
# ==================================================
def plot_phase_mapping(phase_map):
    clusters = list(phase_map.keys())
    phases = list(phase_map.values())

    plt.figure(figsize=(6, 3))
    plt.scatter(clusters, phases, s=120)
    plt.xlabel("Cluster ID")
    plt.ylabel("Damage phase")
    plt.yticks(sorted(set(phases)))
    plt.title("Hierarchical damage phase mapping")
    plt.tight_layout()
    plt.show()


# ==================================================
# MAIN — RUN EVERYTHING
# ==================================================
if __name__ == "__main__":

    # ------------------------------
    # IMPORT YOUR PIPELINE MODULES
    # ------------------------------
    from ae02_preprocessing import preprocess
    from ae04_dimensionality_reduction import reduce_dimensions
    from ae05_clustering import cluster_events
    from ae08_hierarchy_construction import build_hierarchy
    from ae09_damage_trajectories import extract_transitions

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
    # FEATURE DATAFRAME (PHYSICAL)
    # ------------------------------
    feature_df = df[["A", "R", "D", "CNTS", "E", "F"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # ------------------------------
    # HIERARCHY (PHASES)
    # ------------------------------
    phase_map = build_hierarchy(feature_df, labels, n_phases=4)

    # ------------------------------
    # TRANSITIONS
    # ------------------------------
    transitions = extract_transitions(df, labels, time_col=None)

    # ------------------------------
    # PLOTS (ALL 5 FIGURES)
    # ------------------------------
    plot_pca_embedding(X_2d, labels)                           # Figure 1
    plot_cluster_feature_profiles(feature_df, labels)          # Figure 2
    plot_radar_signatures(feature_df, labels)                  # Figure 3
    plot_transition_matrix(transitions, len(np.unique(labels)))# Figure 4
    plot_phase_mapping(phase_map)                               # Figure 5
