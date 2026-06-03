"""
Trains all 5 folds of both the 1D and 2D CNN on GPU using PyTorch.
Architecture mirrors train_keras_1d.py / train_keras_2d.py exactly.

Results saved to output/sunflower/:
  1d/fold_{k}.pt   — PyTorch state dicts
  2d/fold_{k}.pt
  benchmark.csv    — per-fold accuracy + wall-clock training time
"""

import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn.metrics as skm

from src.training.train_keras_1d import load_and_preprocess as load_1d
from src.training.train_keras_2d import load_and_preprocess as load_2d

SUNFLOWER = Path("output/sunflower")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-4
MAX_EPOCHS = 1000
PATIENCE = 100


# ---------- models ----------

class CNN1D(nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(1, 64, 3), nn.ReLU(),
            nn.Conv1d(64, 64, 3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3), nn.ReLU(),
            nn.Conv1d(64, 64, 3), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            flat = self.convs(torch.zeros(1, 1, input_len)).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.head(self.convs(x))


class CNN2D(nn.Module):
    def __init__(self, h, w):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(2, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.convs(torch.zeros(1, 2, h, w)).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.head(self.convs(x))


# ---------- fold splitting ----------

def get_fold(X, y, k):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for i, (tr, te) in enumerate(skf.split(X, y), start=1):
        if i == k:
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_tr, y_tr, random_state=0, test_size=0.3)
            return X_tr, X_va, X_te, y_tr, y_va, y_te


def make_loader(X, y, shuffle=False):
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=shuffle)


# ---------- training ----------

def train_fold(model, X_tr, X_va, X_te, y_tr, y_va, y_te, save_path):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader   = make_loader(X_va, y_va)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(X_va)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    elapsed = time.perf_counter() - t0

    model.load_state_dict(best_state)
    model.eval()
    test_loader = make_loader(X_te, y_te)
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
    y_pred = torch.cat(preds).numpy()
    acc = skm.accuracy_score(y_te, y_pred)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, save_path)
    return acc, elapsed, epoch - PATIENCE  # epoch stopped at


# ---------- main ----------

def main():
    print(f"Device: {DEVICE}")

    print("\nLoading 1D spectra...")
    X_1d_raw, y = load_1d()
    # (samples, timesteps, 1) → (samples, 1, timesteps)  [PyTorch channels-first]
    X_1d = np.transpose(np.squeeze(X_1d_raw, axis=-1)[:, np.newaxis, :], (0, 1, 2))

    print("Loading 2D GAF images...")
    X_2d_raw, _ = load_2d()
    # (samples, H, W, 2) → (samples, 2, H, W)
    X_2d = np.transpose(X_2d_raw, (0, 3, 1, 2))

    h, w = X_2d.shape[2], X_2d.shape[3]
    input_len = X_1d.shape[2]
    print(f"1D input length: {input_len}   2D image: {h}×{w}×2")

    rows = []
    for k in range(1, 6):
        print(f"\n=== Fold {k} ===")

        splits_1d = get_fold(X_1d, y, k)
        splits_2d = get_fold(X_2d, y, k)

        print(f"  Training 1D CNN fold {k}...")
        acc_1d, t_1d, ep_1d = train_fold(
            CNN1D(input_len), *splits_1d,
            save_path=SUNFLOWER / "1d" / f"fold_{k}.pt")
        print(f"  1D fold {k}: acc={acc_1d:.4f}  time={t_1d:.1f}s  stopped ep~{ep_1d}")

        print(f"  Training 2D CNN fold {k}...")
        acc_2d, t_2d, ep_2d = train_fold(
            CNN2D(h, w), *splits_2d,
            save_path=SUNFLOWER / "2d" / f"fold_{k}.pt")
        print(f"  2D fold {k}: acc={acc_2d:.4f}  time={t_2d:.1f}s  stopped ep~{ep_2d}")

        rows.append(dict(fold=k,
                         acc_1d=f"{acc_1d:.4f}", train_time_1d_s=f"{t_1d:.1f}",
                         acc_2d=f"{acc_2d:.4f}", train_time_2d_s=f"{t_2d:.1f}"))

    # Save CSV
    SUNFLOWER.mkdir(parents=True, exist_ok=True)
    csv_path = SUNFLOWER / "benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "acc_1d", "train_time_1d_s", "acc_2d", "train_time_2d_s"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    accs_1d  = [float(r["acc_1d"])          for r in rows]
    accs_2d  = [float(r["acc_2d"])          for r in rows]
    times_1d = [float(r["train_time_1d_s"]) for r in rows]
    times_2d = [float(r["train_time_2d_s"]) for r in rows]

    print("\n" + "=" * 62)
    print(f"{'Fold':<6} {'Acc 1D':>8} {'Time 1D':>12} {'Acc 2D':>8} {'Time 2D':>12}")
    print("-" * 62)
    for r in rows:
        print(f"{r['fold']:<6} {r['acc_1d']:>8} {r['train_time_1d_s']:>11}s {r['acc_2d']:>8} {r['train_time_2d_s']:>11}s")
    print("-" * 62)
    print(f"{'Mean':<6} {np.mean(accs_1d):>8.4f} {np.mean(times_1d):>11.1f}s {np.mean(accs_2d):>8.4f} {np.mean(times_2d):>11.1f}s")
    print("=" * 62)
    print(f"\nResults -> {csv_path}")
    print(f"Models  -> {SUNFLOWER / '1d'}  and  {SUNFLOWER / '2d'}")


if __name__ == "__main__":
    main()
