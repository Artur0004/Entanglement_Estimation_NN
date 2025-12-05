import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to your CSV (2-qubit sweep)
csv_path = r"C:\Users\artxm\Desktop\F_sweep_results_20251121_204541.csv"
df = pd.read_csv(csv_path)

# Paper accuracies (R²) from Pan et al.
paper_r2_f1 = 0.91
paper_r2_f2 = 0.98

# -------------------------------------------------------
# Helpers: parse hidden_sizes and build nice labels
# -------------------------------------------------------
def parse_hidden_sizes(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def pretty_arch(h_list):
    """[256,256,256] -> '3×256', [128,64] -> '128–64'."""
    if not h_list:
        return "?"
    if len(set(h_list)) == 1:
        return f"{len(h_list)}×{h_list[0]}"
    return "–".join(str(h) for h in h_list)

df["hidden_list"] = df["hidden_sizes"].apply(parse_hidden_sizes)
df["arch_pretty"] = df["hidden_list"].apply(pretty_arch)

# -------------------------------------------------------
# Plot function
# -------------------------------------------------------
def plot_architectures_for_dataset(df, dataset_name, paper_r2=None):
    """
    Horizontal bar plot of test_r2 by architecture for one dataset (F1 or F2),
    with a vertical line for the paper's R².
    """
    # Filter and sort (descending so best is first)
    sub = (
        df[df["dataset"] == dataset_name]
        .sort_values("test_r2", ascending=False)
        .reset_index(drop=True)
    )

    if sub.empty:
        print(f"No rows for dataset {dataset_name}")
        return

    labels = sub["arch_pretty"].tolist()
    r2_values = sub["test_r2"].tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(sub))
    ax.barh(y_pos, r2_values, align="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # best at top

    ax.set_xlabel("Test R²")
    ax.set_title(f"2-qubit {dataset_name}: architectures vs Test R²")

    # Paper reference line
    all_vals = r2_values.copy()
    if paper_r2 is not None:
        ax.axvline(paper_r2,color="red", linestyle="--", linewidth=1.5,
                   label=f"Paper R² = {paper_r2:.2f}")
        all_vals.append(paper_r2)

    # Annotate each bar with its R²
    for i, r2 in enumerate(r2_values):
        ax.text(r2 + 0.01, i, f"{r2:.3f}",
                va="center", ha="left", fontsize=8)

    # Tight x-limits for comparison
    xmin = max(0.0, min(all_vals) - 0.05)
    xmax = min(1.0, max(all_vals) + 0.05)
    ax.set_xlim(xmin, xmax)

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    if paper_r2 is not None:
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------
# Make plots
# -------------------------------------------------------
plot_architectures_for_dataset(df, "F1", paper_r2=paper_r2_f1)
plot_architectures_for_dataset(df, "F2", paper_r2=paper_r2_f2)
