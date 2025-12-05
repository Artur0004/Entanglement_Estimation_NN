#!/usr/bin/env python3
"""
Train MLPs on noisy measurement dataset:

- Data path:
    /zhome/b8/7/168711/dataset_noisy_measurements.csv

- Tasks:
    * F2 :  (meas_1..10, mask_1..10) -> (F2_xx..F2_zz)   [9-dim regression]
    * C  :  (F2_xx..F2_zz)            -> Concurrence      [1-dim regression]

For each task and architecture:
    * Train MLP with EARLY STOPPING on Val MSE
    * Save:
        - log.txt
        - best_model.pth
        - run_config.json
        - test_metrics.json

Also writes a global CSV with one row per (task, architecture).

Usage example on HPC:
    python noisy_F2_and_C_sweep.py --tasks F2 C --epochs 100 --batch_size 1024 --lr 1e-3 --patience 15
"""

import os
import csv
import json
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
DATA_PATH_NOISY = os.path.join(ZHOME_BASE, "dataset_noisy_measurements.csv")

RESULTS_BASE_PATH = os.path.join(ZHOME_BASE, "quantum_nn_results_noisy")
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
    def __init__(self, input_dim: int, hidden_sizes, output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))  # regression output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # shape [B, output_dim] or [B, 1]
# ----------------------------------------------


# --------------- DATA HELPERS ------------------
def load_dataset(task: str, logger: Logger):
    """
    Load dataset_noisy_measurements.csv and return (X, y, feature_cols, target_cols).

    task == 'F2':
        X = [meas_1..10, mask_1..10]   (20-dim)
        y = [F2_xx..F2_zz]             (9-dim)

    task == 'C':
        X = [F2_xx..F2_zz]             (9-dim)
        y = Concurrence                (1-dim)
    """
    if not os.path.exists(DATA_PATH_NOISY):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH_NOISY}")

    logger.log(f"Loading noisy dataset from: {DATA_PATH_NOISY}")
    df = pd.read_csv(DATA_PATH_NOISY)

    # Columns present in your dataset
    meas_cols = [c for c in df.columns if c.startswith("meas_")]
    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    f2_cols = [c for c in df.columns if c.startswith("F2_")]

    if "Concurrence" not in df.columns:
        raise KeyError("'Concurrence' column not found in dataset_noisy_measurements.csv.")

    if task == "F2":
        feature_cols = meas_cols + mask_cols
        target_cols = f2_cols
    elif task == "C":
        feature_cols = f2_cols
        target_cols = ["Concurrence"]
    else:
        raise ValueError(f"Unknown task {task}, expected 'F2' or 'C'.")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    logger.log(f"Task {task}: X shape = {X.shape}, y shape = {y.shape}")
    logger.log(f"Feature columns ({len(feature_cols)}): {feature_cols}")
    logger.log(f"Target columns ({len(target_cols)}): {target_cols}")

    return X, y, feature_cols, target_cols


def make_dataloaders(X, y, batch_size: int, logger: Logger):
    """
    Split X, y into train/val/test, standardize X, and create DataLoaders.

    Train/val/test = 70/15/15
    """

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
    y_true, y_pred: tensors with shape [N, d] or [N].
    Returns dict with MSE, MAE, RMSE, R2 computed over all elements.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten over all dims
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    mse = float(np.mean((y_true_flat - y_pred_flat) ** 2))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
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
    task: str,
    arch_name: str,
    hidden_sizes,
    args,
    timestamp: str,
):
    """
    Train one model for given task and architecture.
    Uses EARLY STOPPING on validation MSE.
    Returns dict summarizing run (for global CSV).
    """

    # ---- Create results dir & logger ----
    run_name = f"{task}_{arch_name}_{timestamp}"
    run_dir = os.path.join(RESULTS_BASE_PATH, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = Logger(os.path.join(run_dir, "log.txt"))
    logger.log(f"--- Starting run: {run_name} ---")
    logger.log(f"Task: {task}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # ---- Load data ----
    X, y, feature_cols, target_cols = load_dataset(task, logger)
    train_loader, val_loader, test_loader, scaler = make_dataloaders(
        X, y, batch_size=args.batch_size, logger=logger
    )

    input_dim = len(feature_cols)
    output_dim = len(target_cols)
    logger.log(f"Input dimension: {input_dim}")
    logger.log(f"Output dimension: {output_dim}")
    logger.log(f"Hidden sizes: {hidden_sizes}")
    logger.log(f"Dropout: {args.dropout}")
    logger.log(f"Max epochs: {args.epochs}, Early stopping patience: {args.patience}")

    # ---- Model, loss, optimizer ----
    model = MLP(input_dim=input_dim, hidden_sizes=hidden_sizes,
                output_dim=output_dim, dropout=args.dropout).to(device)
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
        "task": task,
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
        "target_cols": target_cols,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)

    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    logger.log(f"All artifacts saved in: {run_dir}")
    logger.close()

    summary_row = {
        "task": task,
        "arch_name": arch_name,
        "hidden_sizes": str(hidden_sizes),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "run_dir": run_dir,
    }
    return summary_row
# ----------------------------------------------


# ------------------------ MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description="MLP sweep on noisy measurement dataset")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["F2", "C"],
        choices=["F2", "C"],
        help="Which tasks to run: 'F2' (reconstruct Pauli expectations) and/or 'C' (predict concurrence).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in MLP layers")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without val improvement)")
    args = parser.parse_args()

    # A set of architectures to sweep
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
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = os.path.join(RESULTS_BASE_PATH, f"noisy_sweep_results_{timestamp}.csv")

    fieldnames = [
        "task",
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
    for task in args.tasks:
        for arch_name, hidden_sizes in architectures.items():
            row = train_one_architecture(
                task=task,
                arch_name=arch_name,
                hidden_sizes=hidden_sizes,
                args=args,
                timestamp=timestamp,
            )
            rows.append(row)

            # Append to CSV progressively
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
