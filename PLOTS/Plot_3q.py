import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Config: path to your sweep CSV
# -------------------------------------------------------
CSV_PATH = r"C:\Users\artxm\Desktop\3q_F_sweep_results_20251128_205818.csv"

# -------------------------------------------------------
# 1) Load and inspect
# -------------------------------------------------------
df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns.tolist())
print("\nHead:")
print(df.head())

# -------------------------------------------------------
# 2) Parse hidden_sizes string -> list, and derive depth / total units
# -------------------------------------------------------
def parse_hidden_sizes(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

df["hidden_list"] = df["hidden_sizes"].apply(parse_hidden_sizes)
df["depth"] = df["hidden_list"].apply(len)
df["total_units"] = df["hidden_list"].apply(sum)

def format_arch(hidden):
    """Pretty string for architecture, e.g. [256,256,256,256] -> '4×256', [128,64] -> '128–64'."""
    if not hidden:
        return "?"
    if len(set(hidden)) == 1:
        return f"{len(hidden)}×{hidden[0]}"
    return "–".join(str(h) for h in hidden)

df["arch_pretty"] = df["hidden_list"].apply(format_arch)

print("\nExample parsed architectures:")
print(df[["dataset", "arch_name", "hidden_sizes", "arch_pretty", "depth", "total_units"]].head())

# -------------------------------------------------------
# 3) Show best architectures per dataset (F1 vs F2)
# -------------------------------------------------------
for ds in ["F1", "F2"]:
    sub = df[df["dataset"] == ds].copy()
    if sub.empty:
        print(f"\nNo rows for dataset {ds}")
        continue

    sub_sorted = sub.sort_values("test_mse")
    print(f"\n=== Best architectures for dataset {ds} (sorted by test MSE) ===")
    print(
        sub_sorted[
            ["arch_name", "arch_pretty", "hidden_sizes", "test_mse", "test_mae", "test_rmse", "test_r2"]
        ].head(5)
    )

# -------------------------------------------------------
# 4) Plot helpers
# -------------------------------------------------------
def plot_bar_metric_per_arch(df_subset, metric, title, xlabel, save_name=None):
    """
    Horizontal bar plot of a metric vs architecture (pretty label) for a given dataset.
    """
    # Sort by metric (ascending if error, descending if R²)
    ascending = metric != "test_r2"
    df_sorted = df_subset.sort_values(metric, ascending=ascending)

    values = df_sorted[metric].values
    labels = df_sorted["arch_pretty"].values
    y = np.arange(len(df_sorted))

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    ax.barh(y, values)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # best at top
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)

    # annotate each bar with its numeric value
    for i, v in enumerate(values):
        ax.text(
            v + (0.0005 if metric != "test_r2" else 0.005),  # small offset to the right
            i,
            f"{v:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=300)
    plt.show()

# -------------------------------------------------------
# 5) Bar plots: test MSE per architecture (F1, F2)
# -------------------------------------------------------
# Test MSE
for ds in ["F1", "F2"]:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue
    plot_bar_metric_per_arch(
        sub,
        metric="test_mse",
        title=f"3-qubit {ds}: Test MSE by architecture",
        xlabel="Test MSE",
        save_name=f"3q_{ds}_test_mse_per_arch.png",
    )

# Test R²
for ds in ["F1", "F2"]:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue
    plot_bar_metric_per_arch(
        sub,
        metric="test_r2",
        title=f"3-qubit {ds}: Test R² by architecture",
        xlabel="Test R²",
        save_name=f"3q_{ds}_test_r2_per_arch.png",
    )


# -------------------------------------------------------
# 7) Scatter: model size vs performance (R² vs total_units)
# -------------------------------------------------------
plt.figure(figsize=(6, 5))
markers = {"F1": "o", "F2": "s"}
colors = {"F1": "tab:blue", "F2": "tab:orange"}

for ds in ["F1", "F2"]:
    sub = df[df["dataset"] == ds]
    if sub.empty:
        continue
    plt.scatter(
        sub["total_units"],
        sub["test_r2"],
        label=f"{ds} (features {ds})",
        marker=markers.get(ds, "o"),
        alpha=0.8,
        s=40,
    )

plt.xlabel("Total hidden units (sum over layers)")
plt.ylabel("Test R²")
plt.title("3-qubit: Model size vs predictive performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("3q_model_size_vs_r2.png", dpi=300)
plt.show()

print("\nDone. PNG files were saved in the current working directory.")
