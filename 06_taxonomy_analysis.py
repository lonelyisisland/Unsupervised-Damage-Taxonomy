import numpy as np
import pandas as pd

from ae02_preprocessing import preprocess
from ae04_dimensionality_reduction import reduce_dimensions
from ae05_clustering import cluster_events


# --------------------------------------------------
# Cluster-wise taxonomy statistics
# --------------------------------------------------
def summarize_clusters(feature_df, labels):
    """
    Compute physically meaningful statistics for each cluster.
    """

    summaries = {}

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # skip noise if any

        cluster_df = feature_df[labels == cluster_id]

        summaries[int(cluster_id)] = {
            "count": len(cluster_df),
            "mean_energy": cluster_df["E"].mean(),
            "mean_amplitude": cluster_df["A"].mean(),
            "mean_frequency": cluster_df["F"].mean(),
            "mean_duration": cluster_df["D"].mean(),
            "mean_counts": cluster_df["CNTS"].mean(),
        }

    return summaries


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    # -----------------------------
    # 1. Load dataset (IMPORTANT)
    # -----------------------------
    df = pd.read_csv("Mendeley_ae.csv", sep=";")

    # Sanity check (never skip)
    assert "Damage Mechanism" in df.columns, df.columns

    # -----------------------------
    # 2. Preprocess
    # -----------------------------
    X_scaled, y = preprocess(df)

    # Keep ORIGINAL physical features for interpretation
    feature_df = df[["A", "R", "D", "CNTS", "E", "F"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # -----------------------------
    # 3. Dimensionality reduction
    # -----------------------------
    X_2d, _, _ = reduce_dimensions(X_scaled)

    # -----------------------------
    # 4. Clustering
    # -----------------------------
    labels, _ = cluster_events(X_2d, n_clusters=5)

    # -----------------------------
    # 5. Taxonomy analysis
    # -----------------------------
    summaries = summarize_clusters(feature_df, labels)

    # -----------------------------
    # 6. Print results
    # -----------------------------
    print("\n===== AE DAMAGE TAXONOMY SUMMARY =====\n")

    for cid, stats in summaries.items():
        print(f"Cluster {cid}")
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print()
