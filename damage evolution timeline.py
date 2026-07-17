import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ==================================================
# DAMAGE EVOLUTION MAP — TIMELINE
# ==================================================
def plot_damage_evolution_timeline(labels, phase_map=None):
    """
    Event index vs damage state evolution.
    If phase_map is given, plots phase-level evolution.
    """

    events = np.arange(len(labels))

    if phase_map is not None:
        states = np.array([phase_map[int(c)] for c in labels])
        ylabel = "Damage Phase"
        title = "Damage Evolution Map (Phase-level)"
    else:
        states = labels
        ylabel = "Damage Cluster"
        title = "Damage Evolution Map (Cluster-level)"

    plt.figure(figsize=(10, 4))
    plt.scatter(events, states, c=states, cmap="tab10", s=6)
    plt.xlabel("Event index (fatigue progression)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==================================================
# DAMAGE EVOLUTION MAP — TRANSITION NETWORK
# ==================================================
def plot_damage_evolution_network(transitions, min_count=20):
    """
    Directed state-to-state evolution map based on transition frequency.
    """

    states = sorted(
        set([i for i, j in transitions.keys()] +
            [j for i, j in transitions.keys()])
    )

    plt.figure(figsize=(8, 2))

    max_count = max(transitions.values())

    for (i, j), count in transitions.items():
        if count < min_count:
            continue

        plt.arrow(
            i, 0,
            j - i, 0,
            head_width=0.05,
            length_includes_head=True,
            alpha=count / max_count
        )

        plt.text((i + j) / 2, 0.04, str(count),
                 ha="center", fontsize=8)

    plt.scatter(states, [0]*len(states), s=200)

    for s in states:
        plt.text(s, -0.08, f"State {s}", ha="center")

    plt.axis("off")
    plt.title("Damage Evolution Map (State Transitions)")
    plt.tight_layout()
    plt.show()


# ==================================================
# TRANSITION EXTRACTION
# ==================================================
def extract_transitions(labels):
    transitions = []
    for i in range(len(labels) - 1):
        if labels[i] != labels[i + 1]:
            transitions.append((labels[i], labels[i + 1]))
    return Counter(transitions)


# ==================================================
# MAIN (RUN THIS)
# ==================================================
if __name__ == "__main__":

    # ----------------------------------------------
    # IMPORT FROM YOUR EXISTING PIPELINE
    # ----------------------------------------------
    from ae02_preprocessing import preprocess
    from ae04_dimensionality_reduction import reduce_dimensions
    from ae05_clustering import cluster_events
    from ae08_hierarchy_construction import build_hierarchy

    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------
    df = pd.read_csv("Mendeley_ae.csv", sep=";")

    # ----------------------------------------------
    # PREPROCESS
    # ----------------------------------------------
    X_scaled, _ = preprocess(df)

    # ----------------------------------------------
    # DIMENSIONALITY REDUCTION
    # ----------------------------------------------
    X_2d, _, _ = reduce_dimensions(X_scaled)

    # ----------------------------------------------
    # CLUSTERING
    # ----------------------------------------------
    labels, _ = cluster_events(X_2d, n_clusters=5)

    # ----------------------------------------------
    # PHASE (HIERARCHY) MAPPING
    # ----------------------------------------------
    feature_df = df[["A", "R", "D", "CNTS", "E", "F"]].apply(
        pd.to_numeric, errors="coerce"
    )

    phase_map = build_hierarchy(
        feature_df,
        labels,
        n_phases=4
    )

    # ----------------------------------------------
    # TRANSITION EXTRACTION
    # ----------------------------------------------
    transitions = extract_transitions(labels)

    # ----------------------------------------------
    # DAMAGE EVOLUTION MAPS
    # ----------------------------------------------
    # Cluster-level evolution
    plot_damage_evolution_timeline(labels)

    # Phase-level evolution
    plot_damage_evolution_timeline(labels, phase_map=phase_map)

    # Transition network
    plot_damage_evolution_network(transitions, min_count=20)
