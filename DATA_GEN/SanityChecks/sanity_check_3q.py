import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

F1_PATH = r"C:\Users\artxm\Desktop\dataset3_F1.csv"
F2_PATH = r"C:\Users\artxm\Desktop\dataset3_F2.csv"

def check_dataset(path, feature_prefix, name):
    print("\n" + "="*70)
    print(f"Checking dataset: {name}")
    print("="*70)

    df = pd.read_csv(path)
    print(f"File: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # Identify features and label
    label_col = "GlobalEntanglement"
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]

    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"Label column: {label_col}\n")

    X = df[feature_cols]
    y = df[label_col]

    # --- Basic stats ---
    print("=== Feature Statistics (mean, std, min, max) ===")
    print(X.describe().T[['mean', 'std', 'min', 'max']])

    print("\n=== Label Statistics ===")
    print(y.describe())

    # --- Range checks ---
    print("\n=== Range Checks ===")
    # Features should be in [-1,1]
    out_of_bounds = (X.abs() > 1.001).sum()
    if (out_of_bounds > 0).any():
        print("WARNING: Some feature values are outside [-1,1]:")
        print(out_of_bounds[out_of_bounds > 0])
    else:
        print("All feature values within [-1, 1] ✔")

    # Label should be in [0,1]
    below_zero = (y < -1e-6).sum()
    above_one = (y > 1 + 1e-6).sum()
    print(f"Label values < 0 (beyond tiny tolerance): {below_zero}")
    print(f"Label values > 1 (beyond tiny tolerance): {above_one}")
    if below_zero == 0 and above_one == 0:
        print("All label values within [0, 1] (up to numerical noise) ✔")

    # --- Entanglement distribution ---
    print("\n=== GlobalEntanglement Distribution ===")
    zero_count = (y == 0).sum()
    nonzero_count = (y > 0).sum()
    near_zero_count = ((y > 0) & (y < 0.01)).sum()

    print(f"Total samples: {len(df)}")
    print(f"Zero entanglement (E = 0): {zero_count} "
          f"({zero_count/len(df)*100:.2f} %)")
    print(f"Non-zero entanglement (E > 0): {nonzero_count} "
          f"({nonzero_count/len(df)*100:.2f} %)")
    print(f"Near-zero entanglement (0 < E < 0.01): {near_zero_count}")

    # --- Correlation with label ---
    print("\n=== Correlation of each feature with GlobalEntanglement ===")
    corr = df.corr(numeric_only=True)[label_col].drop(label_col)
    print(corr.sort_values(key=lambda s: s.abs(), ascending=False))

    # --- Optional: histogram of label ---
    plt.figure(figsize=(6,4))
    plt.hist(y, bins=100, color='steelblue', edgecolor='black')
    plt.title(f"{name}: Distribution of GlobalEntanglement")
    plt.xlabel("GlobalEntanglement")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_dataset(F1_PATH, feature_prefix="F1_", name="3-qubit F1 dataset")
    check_dataset(F2_PATH, feature_prefix="F2_", name="3-qubit F2 dataset")
