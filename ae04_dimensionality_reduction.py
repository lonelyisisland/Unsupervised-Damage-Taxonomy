import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------
# NumPy-only PCA using SVD
# -----------------------------------
def pca_numpy(X, n_components=10):
    """
    Pure NumPy PCA using SVD.
    Equivalent to sklearn PCA for centered data.
    """
    # Center data
    X_centered = X - np.mean(X, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Project data
    X_pca = np.dot(X_centered, Vt.T[:, :n_components])

    # Explained variance (optional, useful later)
    explained_variance = (S ** 2) / (X.shape[0] - 1)

    return X_pca, Vt[:n_components], explained_variance[:n_components]


# -----------------------------------
# Dimensionality reduction wrapper
# -----------------------------------
def reduce_dimensions(X):
    # Step 1: PCA to 10 components
    X_pca, components, var = pca_numpy(X, n_components=10)

    # Step 2: Use first 2 PCs as embedding (UMAP replacement)
    X_2d = X_pca[:, :2]

    return X_2d, X_pca, components


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    from ae02_preprocessing import preprocess  # your existing NumPy/Pandas-safe preprocessing

    df = pd.read_csv("Mendeley_ae.csv", sep=';')

    X, _ = preprocess(df)
    X = X.values if isinstance(X, pd.DataFrame) else X

    X_2d, _, _ = reduce_dimensions(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=3, alpha=0.6)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA embedding of AE events")
    plt.tight_layout()
    plt.show()
