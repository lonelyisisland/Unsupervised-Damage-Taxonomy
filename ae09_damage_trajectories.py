import pandas as pd
from collections import Counter


# --------------------------------------------------
# Transition extraction
# --------------------------------------------------
def extract_transitions(df, cluster_labels, time_col=None):
    """
    Extract cluster-to-cluster transitions in temporal order.

    If time_col is None, row order is assumed to be acquisition order.
    """

    df_local = df.copy()
    df_local["cluster"] = cluster_labels

    # Remove noise (not expected with k-means, but safe)
    df_local = df_local[df_local["cluster"] != -1]

    # Sort by time if available, else keep original order
    if time_col is not None and time_col in df_local.columns:
        df_local = df_local.sort_values(time_col)
    else:
        # acquisition order = row index
        df_local = df_local.reset_index(drop=True)

    clusters = df_local["cluster"].values

    transitions = []
    for i in range(len(clusters) - 1):
        if clusters[i] != clusters[i + 1]:
            transitions.append((int(clusters[i]), int(clusters[i + 1])))

    return Counter(transitions)


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

    # Dimensionality reduction + clustering
    X_2d, _, _ = reduce_dimensions(X_scaled)
    labels, _ = cluster_events(X_2d, n_clusters=5)

    # Transition analysis
    transitions = extract_transitions(
        df,
        labels,
        time_col=None  # set to column name if true time exists
    )

    print("\nTop cluster transitions:")
    for (c1, c2), count in transitions.most_common(10):
        print(f"  {c1} → {c2}: {count}")
