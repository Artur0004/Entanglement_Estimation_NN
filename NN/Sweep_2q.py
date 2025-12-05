#!/usr/bin/env python3
"""
Sweep many MLP architectures on dataset_F1.csv and dataset_F2.csv.

- Datasets are assumed to live in your zhome:
    /zhome/b8/7/168711/dataset_F1.csv
    /zhome/b8/7/168711/dataset_F2.csv

- For each dataset and architecture:
    * Train MLP for up to N epochs with EARLY STOPPING on Val MSE
    * Keep best model according to validation MSE
    * Evaluate on test set
    * Save:
        - log.txt
        - best_model.pth
        - run_config.json
        - test_metrics.json

- Also writes a global CSV with one row per (dataset, architecture).

Usage examples on HPC:
    python dsnn_F_sweep.py --datasets F1 F2 --epochs 100 --batch_size 1024 --lr 1e-3 --patience 15
"""

import os
import json
import csv
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------- PATHS --------------------
ZHOME_BASE = "/zhome/b8/7/168711"
DATA_PATH_F1 = os.path.join(ZHOME_BASE, "dataset_F1.csv")
DATA_PATH_F2 = os.path.join(ZHOME_BASE, "dataset_F2.csv")

RESULTS_BASE_PATH = os.path.join(ZHOME_BASE, "quantum_nn_results")
os.makedirs(RESULTS_BASE_PATH, exist_ok=True)
# ------------------------------------------------


# ------------------- LOGGER --------------------
class Logger:
    def __init__(self, log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file = open(log_file_path, "w")

    def log(self, msg: str):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()
# ----------------------------------------------


# ------------------ MODEL ----------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))  # regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]
# ----------------------------------------------


# --------------- DATA HELPERS ------------------
def load_dataset(dataset_name: str, logger: Logger):
    """Load F1 or F2 datasets from CSV and return (X, y) as numpy arrays."""
    if dataset_name == "F1":
        path = DATA_PATH_F1
        feature_cols = ["F1_xx", "F1_xy", "F1_xz", "F1_yy", "F1_yz", "F1_zz"]
    elif dataset_name == "F2":
        path = DATA_PATH_F2
        feature_cols = [
            "F2_xx", "F2_xy", "F2_xz",
            "F2_yx", "F2_yy", "F2_yz",
            "F2_zx", "F2_zy", "F2_zz",
        ]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, expected 'F1' or 'F2'.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    logger.log(f"Loading {dataset_name} from: {path}")
    df = pd.read_csv(path)

    # Assume target column is named 'Concurrence'
    if "Concurrence" not in df.columns:
        raise KeyError(f"'Concurrence' column not found in {path}.")

    X = df[feature_cols].values.astype(np.float32)
    y = df["Concurrence"].values.astype(np.float32)
    logger.log(f"{dataset_name}: X shape = {X.shape}, y shape = {y.shape}")

    return X, y, feature_cols


def make_dataloaders(X, y, batch_size: int, logger: Logger):
    """Split X, y into train/val/test, standardize, and create DataLoaders."""

    # Train/val/test = 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, shuffle=True
    )

    logger.log(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # Standardize features using train statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
# ----------------------------------------------


# --------------- METRICS & TRAINING -----------
def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: 1D tensors or numpy arrays.
    Returns dict with MSE, MAE, RMSE, R2.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    # R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds.append(out)
            targets.append(yb)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return compute_metrics(targets, preds)


def train_one_architecture(
    dataset_name: str,
    arch_name: str,
    hidden_sizes,
    args,
    timestamp: str,
):
    """
    Train one model for given dataset and architecture.
    Uses EARLY STOPPING on validation MSE.
    Returns dict summarizing run (for CSV).
    """

    # ---- Create results dir & logger ----
    run_name = f"{dataset_name}_{arch_name}_{timestamp}"
    run_dir = os.path.join(RESULTS_BASE_PATH, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = Logger(os.path.join(run_dir, "log.txt"))
    logger.log(f"--- Starting run: {run_name} ---")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # ---- Load data ----
    X, y, feature_cols = load_dataset(dataset_name, logger)
    train_loader, val_loader, test_loader, scaler = make_dataloaders(
        X, y, batch_size=args.batch_size, logger=logger
    )

    input_dim = len(feature_cols)
    logger.log(f"Input dimension: {input_dim}")
    logger.log(f"Hidden sizes: {hidden_sizes}")
    logger.log(f"Dropout: {args.dropout}")
    logger.log(f"Max epochs: {args.epochs}, Early stopping patience: {args.patience}")

    # ---- Model, loss, optimizer ----
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Train loop with EARLY STOPPING ----
    best_val_mse = float("inf")
    best_epoch = -1
    best_model_path = os.path.join(run_dir, "best_model.pth")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device)
        val_mse = val_metrics["mse"]

        logger.log(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train MSE: {train_loss:.6f}, "
            f"Val MSE: {val_mse:.6f}, "
            f"Val R2: {val_metrics['r2']:.4f}"
        )

        # Track best model by validation MSE
        if val_mse < best_val_mse - 1e-8:  # small tolerance
            best_val_mse = val_mse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"  -> New best model saved at epoch {epoch} with Val MSE {best_val_mse:.6f}")
        else:
            epochs_no_improve += 1
            logger.log(f"  -> No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= args.patience:
                logger.log(f"EARLY STOPPING at epoch {epoch} (patience={args.patience})")
                break

    logger.log(f"Training complete. Best epoch: {best_epoch} (Val MSE={best_val_mse:.6f})")

    # ---- Final test using best model ----
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    logger.log(f"FINAL TEST metrics: {test_metrics}")

    # ---- Save configs & metrics ----
    run_config = {
        "dataset": dataset_name,
        "arch_name": arch_name,
        "hidden_sizes": hidden_sizes,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "patience": args.patience,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "feature_cols": feature_cols,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    logger.log(f"All artifacts saved in: {run_dir}")
    logger.close()

    # This row is what you'll later read as "accuracy" per architecture & dataset
    summary_row = {
        "dataset": dataset_name,
        "arch_name": arch_name,
        "hidden_sizes": str(hidden_sizes),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],  # you can treat R2 as an accuracy-like measure
        "run_dir": run_dir,
    }
    return summary_row
# ----------------------------------------------


# ------------------------ MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description="Sweep MLP architectures on dataset_F1/F2")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["F1", "F2"],
        choices=["F1", "F2"],
        help="Which datasets to run on (default: both F1 and F2)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in MLP layers")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without val improvement)")
    args = parser.parse_args()

    # EXTENDED SET OF ARCHITECTURES
    # (You can remove or add entries to control how many you try)
    architectures = {
        "mlp_64":           [64],
        "mlp_128":          [128],
        "mlp_256":          [256],

        "mlp_64x64":        [64, 64],
        "mlp_128x64":       [128, 64],
        "mlp_128x128":      [128, 128],
        "mlp_256x128":      [256, 128],
        "mlp_256x256":      [256, 256],

        "mlp_3x64":         [64, 64, 64],
        "mlp_3x128":        [128, 128, 128],
        "mlp_3x256":        [256, 256, 256],

        "mlp_4x128":        [128, 128, 128, 128],
        "mlp_4x256":        [256, 256, 256, 256],
    }

    # Global CSV across all datasets & architectures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = os.path.join(RESULTS_BASE_PATH, f"F_sweep_results_{timestamp}.csv")

    fieldnames = [
        "dataset",
        "arch_name",
        "hidden_sizes",
        "best_epoch",
        "best_val_mse",
        "test_mse",
        "test_mae",
        "test_rmse",
        "test_r2",
        "run_dir",
    ]

    rows = []
    for ds in args.datasets:
        for arch_name, hidden_sizes in architectures.items():
            row = train_one_architecture(
                dataset_name=ds,
                arch_name=arch_name,
                hidden_sizes=hidden_sizes,
                args=args,
                timestamp=timestamp,
            )
            rows.append(row)

            # Append to CSV progressively (like train2.py)
            if not os.path.exists(summary_csv_path):
                with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
            else:
                with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nSummary CSV written to: {summary_csv_path}")


if __name__ == "__main__":
    main()
