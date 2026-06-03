"""
Extracts one training sample, corrupts it, then benchmarks inference through
the 1D and 2D CNN ensembles (5 folds each).

Timing columns:
  time_1d_ms        - full 1D pipeline (preprocess + forward) per sample
  time_gaf_ms       - GAF transform only (downsample + GASF/GADF + stack)
  time_2d_model_ms  - 2D forward pass only (after GAF is already computed)
  time_2d_total_ms  - GAF + 2D forward combined

All timings averaged over N_RUNS forward passes with GPU sync.
Results saved to output/sunflower/inference_timing.csv.
"""

import sys
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.training.train_keras_1d import load_and_preprocess as load_1d
from src.utils.gaf_transform import gaf_transform
from scripts.corrupt_spectrum import corrupt

SUNFLOWER  = Path("output/sunflower")
SAMPLE_DIR = SUNFLOWER / "test_sample"
GAF_SIZE   = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RUNS     = 100   # forward passes to average over for stable timing

LABEL_NAMES = {0: "HDPE", 1: "LDPE", 2: "PP", 3: "PS", 4: "PVC", 5: "PET"}


# ---------- model definitions (identical to benchmark_2d_vs_1d_torch.py) ----------

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


# ---------- helpers ----------

def load_fold_models(arch_fn, folder):
    models = []
    for k in range(1, 6):
        m = arch_fn()
        m.load_state_dict(torch.load(folder / f"fold_{k}.pt", map_location=DEVICE))
        m.to(DEVICE).eval()
        models.append(m)
    return models


def ensemble_predict(models, tensor):
    with torch.no_grad():
        probs = torch.stack([torch.softmax(m(tensor), dim=1) for m in models]).mean(0)
    return probs.argmax(1).item(), probs.max().item()


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, n_runs):
    """Run fn() n_runs times with GPU sync, return (result, avg_ms)."""
    fn()                    # warmup
    sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = fn()
    sync()
    avg_ms = (time.perf_counter() - t0) / n_runs * 1000
    return result, avg_ms


def load_spectrum_csv(path):
    ab = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                ab.append(float(row[1]))   # absorbance is column 1
            except (ValueError, IndexError):
                continue
    return np.array(ab, dtype=np.float64)


def normalize(spec):
    lo, hi = spec.min(), spec.max()
    return (spec - lo) / (hi - lo) if hi > lo else spec


def spec_to_gaf_tensor(spec):
    """Downsample → GASF/GADF → (1, 2, 64, 64) tensor on DEVICE."""
    x_old = np.linspace(0, 1, len(spec))
    x_new = np.linspace(0, 1, GAF_SIZE)
    spec_ds = np.interp(x_new, x_old, spec)
    gasf, gadf = gaf_transform(spec_ds)
    img = np.stack([gasf, gadf], axis=0).astype(np.float32)
    return torch.tensor(img).unsqueeze(0).to(DEVICE)


# ---------- main ----------

def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- extract sample 0 ----
    print("Loading training data...")
    absorbance_values, labels = load_1d()
    spectra = np.squeeze(absorbance_values, axis=-1)   # (6000, 1868) already [0,1]

    sample = spectra[0]
    true_label = int(labels[0])
    print(f"Sample 0: label={LABEL_NAMES[true_label]}  points={len(sample)}")

    sample_csv = SAMPLE_DIR / "sample.csv"
    wavenumbers = np.linspace(400, 4000, len(sample))
    with open(sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavenumber", "absorbance"])
        for wn, ab in zip(wavenumbers, sample):
            w.writerow([f"{wn:.2f}", f"{ab:.8f}"])
    print(f"Saved raw sample -> {sample_csv}\n")

    # ---- corrupt ----
    print("Generating corrupted variants:")
    corrupt(str(sample_csv), noise_std=0.01, shift_amount=0.05, seed=0)

    variants = {
        "original":  sample_csv,
        "noisy":     SAMPLE_DIR / "sample_noisy.csv",
        "shifted":   SAMPLE_DIR / "sample_shifted.csv",
        "corrupted": SAMPLE_DIR / "sample_corrupted.csv",
    }

    # ---- load models ----
    print(f"\nLoading models on {DEVICE}...")
    models_1d = load_fold_models(lambda: CNN1D(1868),            SUNFLOWER / "1d")
    models_2d = load_fold_models(lambda: CNN2D(GAF_SIZE, GAF_SIZE), SUNFLOWER / "2d")
    print("5 x 1D + 5 x 2D fold models loaded.\n")

    # ---- inference + timing ----
    header = (
        f"{'Variant':<12} {'True':>6} "
        f"{'1D pred':>8} {'1D conf':>8} {'1D ms':>9} | "
        f"{'2D pred':>8} {'2D conf':>8} {'GAF ms':>9} {'2D ms':>9} {'Total ms':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name, path in variants.items():
        ab = load_spectrum_csv(path)
        spec = normalize(ab)

        # -- 1D --
        t1d_in = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        (pred_1d, conf_1d), t_1d = time_fn(lambda: ensemble_predict(models_1d, t1d_in), N_RUNS)

        # -- GAF transform (CPU) --
        _, t_gaf = time_fn(lambda: spec_to_gaf_tensor(spec), N_RUNS)

        # -- 2D model (pre-convert tensor once so we time model only) --
        t2d_in = spec_to_gaf_tensor(spec)
        (pred_2d, conf_2d), t_2d_model = time_fn(lambda: ensemble_predict(models_2d, t2d_in), N_RUNS)

        t_2d_total = t_gaf + t_2d_model

        print(
            f"{name:<12} {LABEL_NAMES[true_label]:>6} "
            f"{LABEL_NAMES[pred_1d]:>8} {conf_1d:>7.1%} {t_1d:>8.3f}ms | "
            f"{LABEL_NAMES[pred_2d]:>8} {conf_2d:>7.1%} "
            f"{t_gaf:>8.3f}ms {t_2d_model:>8.3f}ms {t_2d_total:>9.3f}ms"
        )
        rows.append(dict(
            variant=name,
            true_label=LABEL_NAMES[true_label],
            pred_1d=LABEL_NAMES[pred_1d], conf_1d=f"{conf_1d:.4f}", time_1d_ms=f"{t_1d:.4f}",
            pred_2d=LABEL_NAMES[pred_2d], conf_2d=f"{conf_2d:.4f}",
            time_gaf_ms=f"{t_gaf:.4f}",
            time_2d_model_ms=f"{t_2d_model:.4f}",
            time_2d_total_ms=f"{t_2d_total:.4f}",
        ))

    # ---- save CSV ----
    out_csv = SUNFLOWER / "inference_timing.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
