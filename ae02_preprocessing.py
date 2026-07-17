import pandas as pd
import numpy as np


def preprocess(df):
    """
    Preprocess AE dataset.
    Returns:
        X : numpy array of features
        y : labels (Damage Mechanism)
    """

    print("Original shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Separate label and features
    y = df["Damage Mechanism"]
    X = df.drop(columns=["Damage Mechanism"])

    # Convert to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    X = X.fillna(X.median())

    # Robust scaling (NumPy only)
    median = np.median(X.values, axis=0)
    q1 = np.percentile(X.values, 25, axis=0)
    q3 = np.percentile(X.values, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0

    X_scaled = (X.values - median) / iqr

    print("Preprocessing complete.")
    print("Processed shape:", X_scaled.shape)

    return X_scaled, y


# --------------------------------------------------
# Only runs if file is executed directly
# --------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("Mendeley_ae.csv", sep=";")
    X, y = preprocess(df)
