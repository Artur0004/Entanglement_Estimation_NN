import pandas as pd
import numpy as np

# --- Paths (adjust if needed) ---
path_data = r"C:\Users\artxm\Desktop\dataset_noisy_measurements.csv"
path_dirs = r"C:\Users\artxm\Desktop\measurement_directions.csv"

# --- Load CSVs ---
df = pd.read_csv(path_data)
dirs = pd.read_csv(path_dirs)

print("=== dataset_noisy_measurements.csv ===")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head())

# Basic column grouping
meas_cols = [c for c in df.columns if c.startswith("meas_")]
mask_cols = [c for c in df.columns if c.startswith("mask_")]
f2_cols   = [c for c in df.columns if c.startswith("F2_")]
conc_col  = "Concurrence"

print("\n#meas cols:", len(meas_cols))
print("#mask cols:", len(mask_cols))
print("#F2 cols:  ", len(f2_cols))

# --- Basic sanity checks ---

# 1) Measurement + mask shapes
assert len(meas_cols) == len(mask_cols), "meas_ and mask_ column counts differ!"

# 2) meas_* should be in [-1,1] (with some tiny numerical tolerance)
meas_values = df[meas_cols].to_numpy()
print("\nmeas_* range: min = %.4f, max = %.4f" % (meas_values.min(), meas_values.max()))
assert meas_values.min() >= -1.0001 and meas_values.max() <= 1.0001, "meas_* out of [-1,1] range!"

# 3) mask_* should be 0 or 1
mask_values = df[mask_cols].to_numpy()
unique_masks = np.unique(mask_values)
print("mask_* unique values:", unique_masks)
assert set(unique_masks).issubset({0, 1}), "mask_* has values other than 0 or 1!"

# 4) F2_* should be in [-1,1]
f2_values = df[f2_cols].to_numpy()
print("F2_* range:  min = %.4f, max = %.4f" % (f2_values.min(), f2_values.max()))
assert f2_values.min() >= -1.0001 and f2_values.max() <= 1.0001, "F2_* out of [-1,1] range!"

# 5) Concurrence in [0,1]
conc_values = df[conc_col].to_numpy()
print("Concurrence range: min = %.4f, max = %.4f" % (conc_values.min(), conc_values.max()))
assert conc_values.min() >= -1e-6 and conc_values.max() <= 1.000001, "Concurrence not in [0,1]!"

print("\n=== measurement_directions.csv ===")
print("Shape:", dirs.shape)
print("Columns:", list(dirs.columns))
print(dirs.head())

# Check Bloch vectors are normalized (approximately length 1)
for side in ["A", "B"]:
    n = dirs[[f"n{side}x", f"n{side}y", f"n{side}z"]].to_numpy()
    norms = np.linalg.norm(n, axis=1)
    print(f"\nBloch norms qubit {side}: min = %.4f, max = %.4f" % (norms.min(), norms.max()))
    assert np.allclose(norms, 1.0, atol=1e-6), f"Bloch vectors for qubit {side} are not normalized!"

print("\nAll sanity checks passed ✔")
