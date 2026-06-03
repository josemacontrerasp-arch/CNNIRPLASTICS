"""
Trains all 5 folds of the 2D CNN, loads (or trains) the 1D CNN folds,
then compares accuracy and wall-clock training time.

Results are written to output/sunflower/:
  1d/fold_{k}.keras   — 1D models (copied/trained here)
  2d/fold_{k}.keras   — 2D models trained fresh
  benchmark.csv       — per-fold accuracy + training time
"""

import sys
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import sklearn.metrics as skm
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from src.training.train_keras_1d import load_and_preprocess as load_1d
from src.training.train_keras_2d import load_and_preprocess as load_2d

SUNFLOWER = Path("output/sunflower")
EXISTING_1D = Path("output/models")


# ---------- model builders (mirror the training scripts exactly) ----------

def _build_1d(shape, seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    inputs = layers.Input(shape)
    x = layers.Conv1D(64, 3, activation='relu')(inputs)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.MaxPool1D()(x)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.MaxPool1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(6, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="cnn1d")


def _build_2d(shape, seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    inputs = layers.Input(shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(6, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="cnn2d")


# ---------- fold splitter (identical parameters to both training scripts) ----------

def _get_fold_splits(X, y, k):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        if i == k:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, random_state=0, test_size=0.3)
            return X_train, X_val, X_test, y_train, y_val, y_test
    raise ValueError(f"Fold {k} not found")


def _early_stop():
    return keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, mode="min", restore_best_weights=True)


def _compile(model):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["acc"])


# ---------- per-fold routines ----------

def run_1d_fold(k, X, y, save_dir):
    """Load existing 1D model if available, otherwise train it. Return (accuracy, train_time_s)."""
    existing = EXISTING_1D / f"fold_{k}.keras"
    out_path = save_dir / f"fold_{k}.keras"

    if existing.exists():
        model = keras.models.load_model(existing)
        _, _, X_test, _, _, y_test = _get_fold_splits(X, y, k)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        acc = skm.accuracy_score(y_test, y_pred)
        model.save(out_path)
        print(f"  1D fold {k}: loaded from {existing}  acc={acc:.4f}")
        return acc, None   # training time unavailable for pre-trained models

    # Fold not pre-trained — train from scratch
    X_train, X_val, X_test, y_train, y_val, y_test = _get_fold_splits(X, y, k)
    model = _build_1d(shape=X_train.shape[1:])
    _compile(model)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=1000, batch_size=64, shuffle=True, verbose=0,
              callbacks=[_early_stop()])
    elapsed = time.perf_counter() - t0
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = skm.accuracy_score(y_test, y_pred)
    model.save(out_path)
    print(f"  1D fold {k}: trained  acc={acc:.4f}  time={elapsed:.1f}s")
    return acc, elapsed


def run_2d_fold(k, X, y, save_dir):
    """Train 2D CNN fold k from scratch. Return (accuracy, train_time_s)."""
    X_train, X_val, X_test, y_train, y_val, y_test = _get_fold_splits(X, y, k)
    model = _build_2d(shape=X_train.shape[1:])
    _compile(model)
    t0 = time.perf_counter()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=1000, batch_size=64, shuffle=True, verbose=0,
              callbacks=[_early_stop()])
    elapsed = time.perf_counter() - t0
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = skm.accuracy_score(y_test, y_pred)
    model.save(save_dir / f"fold_{k}.keras")
    print(f"  2D fold {k}: trained  acc={acc:.4f}  time={elapsed:.1f}s")
    return acc, elapsed


# ---------- main ----------

def main():
    dir_1d = SUNFLOWER / "1d"
    dir_2d = SUNFLOWER / "2d"
    dir_1d.mkdir(parents=True, exist_ok=True)
    dir_2d.mkdir(parents=True, exist_ok=True)

    print("Loading 1D spectra...")
    X_1d, y_1d = load_1d()

    print("Loading 2D GAF images (from cache or computing)...")
    X_2d, y_2d = load_2d()

    rows = []
    for k in range(1, 6):
        print(f"\n=== Fold {k} ===")
        acc_1d, time_1d = run_1d_fold(k, X_1d, y_1d, dir_1d)
        acc_2d, time_2d = run_2d_fold(k, X_2d, y_2d, dir_2d)
        rows.append(dict(fold=k,
                         acc_1d=acc_1d,
                         train_time_1d_s="" if time_1d is None else f"{time_1d:.1f}",
                         acc_2d=acc_2d,
                         train_time_2d_s=f"{time_2d:.1f}"))

    # Save CSV
    csv_path = SUNFLOWER / "benchmark.csv"
    fieldnames = ["fold", "acc_1d", "train_time_1d_s", "acc_2d", "train_time_2d_s"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Fold':<6} {'Acc 1D':>8} {'Time 1D':>12} {'Acc 2D':>8} {'Time 2D':>12}")
    print("-" * 60)
    for r in rows:
        t1 = r["train_time_1d_s"] if r["train_time_1d_s"] else "pre-trained"
        print(f"{r['fold']:<6} {r['acc_1d']:>8.4f} {t1:>12} {r['acc_2d']:>8.4f} {r['train_time_2d_s']:>11}s")

    accs_1d  = [r["acc_1d"]  for r in rows]
    accs_2d  = [r["acc_2d"]  for r in rows]
    times_2d = [float(r["train_time_2d_s"]) for r in rows]
    times_1d = [float(r["train_time_1d_s"]) for r in rows if r["train_time_1d_s"]]

    print("-" * 60)
    t1_summary = f"{np.mean(times_1d):.1f}s" if times_1d else "pre-trained"
    print(f"{'Mean':<6} {np.mean(accs_1d):>8.4f} {t1_summary:>12} {np.mean(accs_2d):>8.4f} {np.mean(times_2d):>11.1f}s")
    print("=" * 60)
    print(f"\nBenchmark saved to {csv_path}")
    print(f"2D models saved to {dir_2d}")
    print(f"1D models saved to {dir_1d}")


if __name__ == "__main__":
    main()
