import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Load sweep CSV
# ---------------------------
csv_path = r"C:\Users\artxm\Desktop\quantum_nn_results_noisy\noisy_sweep_results_20251126_214112.csv"
df = pd.read_csv(csv_path)

# Ensure numeric
num_cols = ["best_epoch", "best_val_mse", "test_mse",
            "test_mae", "test_rmse", "test_r2"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------------------------
# Helpers: parse architectures + pretty labels
# ---------------------------
def parse_hidden_sizes(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def pretty_arch(h_list):
    """Turn [256,256,256] -> '3×256', [128,64] -> '128–64'."""
    if not h_list:
        return "?"
    if len(set(h_list)) == 1:
        return f"{len(h_list)}×{h_list[0]}"
    return "–".join(str(h) for h in h_list)

if "hidden_sizes" in df.columns:
    df["hidden_list"] = df["hidden_sizes"].apply(parse_hidden_sizes)
    df["arch_pretty"] = df["hidden_list"].apply(pretty_arch)
else:
    df["hidden_list"] = [[] for _ in range(len(df))]
    df["arch_pretty"] = df["arch_name"]

# ---------------------------
# Plot helper
# ---------------------------
def plot_task_metrics(df_task, task_name, save_prefix=None):
    """
    Make two nice plots for one task:
      - Test R² per architecture
      - Test MSE per architecture
    """
    if df_task.empty:
        print(f"No rows for task {task_name}")
        return

    # ---- R² (higher is better) ----
    df_r2 = df_task.sort_values("test_r2", ascending=False).reset_index(drop=True)
    y = np.arange(len(df_r2))
    r2_vals = df_r2["test_r2"].values
    labels = df_r2["arch_pretty"].values

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    ax.barh(y, r2_vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel("Test R²")
    ax.set_title(f"{task_name}: Test R² by architecture")
    ax.grid(axis="x", alpha=0.3)

    # annotate
    for i, v in enumerate(r2_vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_R2.png", dpi=300)
    plt.show()

    # ---- MSE (lower is better) ----
    df_mse = df_task.sort_values("test_mse", ascending=True).reset_index(drop=True)
    y = np.arange(len(df_mse))
    mse_vals = df_mse["test_mse"].values
    labels = df_mse["arch_pretty"].values

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    ax.barh(y, mse_vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE")
    ax.set_title(f"{task_name}: Test MSE by architecture")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(mse_vals):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_MSE.png", dpi=300)
    plt.show()

# ---------------------------
# F2 reconstruction task
# ---------------------------
df_F2 = df[df["task"] == "F2"].copy()
plot_task_metrics(df_F2,
                  task_name="F2 reconstruction (noisy → clean Pauli expectations)",
                  save_prefix="noisy_F2_reconstruction")

# ---------------------------
# Concurrence prediction task
# ---------------------------
df_C = df[df["task"] == "C"].copy()
plot_task_metrics(df_C,
                  task_name="Concurrence prediction (F2 → C)",
                  save_prefix="noisy_C_prediction")

print("Done – plots saved as PNGs in the current directory.")
