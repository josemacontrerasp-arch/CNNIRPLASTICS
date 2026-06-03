"""
Picks one sample per class (HDPE, LDPE, PP, PS, PVC, PET),
corrupts each, then benchmarks 1D vs 2D inference.

Timing columns (all averaged over N_RUNS GPU-synced passes):
  time_1d_ms        - preprocess + 1D ensemble forward
  time_gaf_ms       - GAF transform only (downsample + GASF/GADF)
  time_2d_model_ms  - 2D ensemble forward only
  time_2d_total_ms  - GAF + 2D forward

Outputs:
  output/sunflower/test_samples/   - per-class sample + corrupted CSVs
  output/sunflower/inference_timing_extended.csv
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
SAMPLE_DIR = SUNFLOWER / "test_samples"
GAF_SIZE   = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_RUNS     = 100

LABEL_NAMES = {0: "HDPE", 1: "LDPE", 2: "PP", 3: "PS", 4: "PVC", 5: "PET"}
VARIANTS    = ["original", "noisy", "shifted", "corrupted"]


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


def timed(fn, n_runs):
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = fn()
    sync()
    return result, (time.perf_counter() - t0) / n_runs * 1000


def normalize(spec):
    lo, hi = spec.min(), spec.max()
    return (spec - lo) / (hi - lo) if hi > lo else spec


def load_spectrum_csv(path):
    ab = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            try:
                ab.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.array(ab, dtype=np.float64)


def spec_to_gaf_tensor(spec):
    x_old = np.linspace(0, 1, len(spec))
    x_new = np.linspace(0, 1, GAF_SIZE)
    spec_ds = np.interp(x_new, x_old, spec)
    gasf, gadf = gaf_transform(spec_ds)
    img = np.stack([gasf, gadf], axis=0).astype(np.float32)
    return torch.tensor(img).unsqueeze(0).to(DEVICE)


def run_variant(path, models_1d, models_2d):
    ab   = load_spectrum_csv(path)
    spec = normalize(ab)

    t1d_in = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    (pred_1d, conf_1d), t_1d = timed(lambda: ensemble_predict(models_1d, t1d_in), N_RUNS)

    _, t_gaf = timed(lambda: spec_to_gaf_tensor(spec), N_RUNS)

    t2d_in = spec_to_gaf_tensor(spec)
    (pred_2d, conf_2d), t_2d = timed(lambda: ensemble_predict(models_2d, t2d_in), N_RUNS)

    return pred_1d, conf_1d, t_1d, pred_2d, conf_2d, t_gaf, t_2d


# ---------- main ----------

def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    absorbance_values, labels = load_1d()
    spectra = np.squeeze(absorbance_values, axis=-1)

    # Pick the first sample encountered for each class
    class_samples = {}
    for i, lbl in enumerate(labels):
        if lbl not in class_samples:
            class_samples[lbl] = i
        if len(class_samples) == 6:
            break

    print(f"Picked one sample per class: { {LABEL_NAMES[k]: v for k, v in class_samples.items()} }\n")

    # Save and corrupt each sample
    sample_paths = {}   # label -> {variant: Path}
    wavenumbers = np.linspace(400, 4000, spectra.shape[1])

    for lbl, idx in class_samples.items():
        name = LABEL_NAMES[lbl]
        sample_csv = SAMPLE_DIR / f"{name}_sample.csv"
        with open(sample_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wavenumber", "absorbance"])
            for wn, ab in zip(wavenumbers, spectra[idx]):
                w.writerow([f"{wn:.2f}", f"{ab:.8f}"])

        print(f"  {name} (idx {idx}): corrupting...")
        corrupt(str(sample_csv), noise_std=0.01, shift_amount=0.05, seed=0)

        sample_paths[lbl] = {
            "original":  sample_csv,
            "noisy":     SAMPLE_DIR / f"{name}_sample_noisy.csv",
            "shifted":   SAMPLE_DIR / f"{name}_sample_shifted.csv",
            "corrupted": SAMPLE_DIR / f"{name}_sample_corrupted.csv",
        }

    # Load models
    print(f"\nLoading models on {DEVICE}...")
    models_1d = load_fold_models(lambda: CNN1D(1868),                SUNFLOWER / "1d")
    models_2d = load_fold_models(lambda: CNN2D(GAF_SIZE, GAF_SIZE),  SUNFLOWER / "2d")
    print("5 x 1D + 5 x 2D fold models loaded.\n")

    # Header
    hdr = (f"{'Class':<6} {'Variant':<10} {'True':>5} "
           f"{'1D':>6} {'1D%':>6} {'1D ms':>8} | "
           f"{'2D':>6} {'2D%':>6} {'GAF ms':>8} {'2D ms':>8} {'Tot ms':>8} "
           f"{'OK?':>5}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    correct_1d = correct_2d = 0
    total = 0

    for lbl in sorted(class_samples):
        name = LABEL_NAMES[lbl]
        for variant in VARIANTS:
            path = sample_paths[lbl][variant]
            p1d, c1d, t1d, p2d, c2d, tgaf, t2d = run_variant(path, models_1d, models_2d)
            ok1 = "Y" if p1d == lbl else "N"
            ok2 = "Y" if p2d == lbl else "N"
            total += 1
            correct_1d += p1d == lbl
            correct_2d += p2d == lbl

            print(
                f"{name:<6} {variant:<10} {LABEL_NAMES[lbl]:>5} "
                f"{LABEL_NAMES[p1d]:>6} {c1d:>5.1%} {t1d:>7.3f}ms | "
                f"{LABEL_NAMES[p2d]:>6} {c2d:>5.1%} {tgaf:>7.3f}ms {t2d:>7.3f}ms {tgaf+t2d:>7.3f}ms "
                f"{'1D:'+ok1+' 2D:'+ok2:>8}"
            )
            rows.append(dict(
                class_name=name, variant=variant, true_label=LABEL_NAMES[lbl],
                pred_1d=LABEL_NAMES[p1d], conf_1d=f"{c1d:.4f}", time_1d_ms=f"{t1d:.4f}",
                correct_1d=(p1d == lbl),
                pred_2d=LABEL_NAMES[p2d], conf_2d=f"{c2d:.4f}",
                time_gaf_ms=f"{tgaf:.4f}", time_2d_model_ms=f"{t2d:.4f}",
                time_2d_total_ms=f"{tgaf+t2d:.4f}", correct_2d=(p2d == lbl),
            ))

    # Summary
    print("-" * len(hdr))
    print(f"\nAccuracy  1D: {correct_1d}/{total} ({correct_1d/total:.1%})   2D: {correct_2d}/{total} ({correct_2d/total:.1%})")

    t1d_vals  = [float(r["time_1d_ms"])    for r in rows]
    tgaf_vals = [float(r["time_gaf_ms"])   for r in rows]
    t2d_vals  = [float(r["time_2d_model_ms"]) for r in rows]
    t2t_vals  = [float(r["time_2d_total_ms"]) for r in rows]
    print(f"Avg times  1D: {np.mean(t1d_vals):.3f}ms   GAF: {np.mean(tgaf_vals):.3f}ms   2D model: {np.mean(t2d_vals):.3f}ms   2D total: {np.mean(t2t_vals):.3f}ms")

    # Save
    out_csv = SUNFLOWER / "inference_timing_extended.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {out_csv}")


if __name__ == "__main__":
    main()
