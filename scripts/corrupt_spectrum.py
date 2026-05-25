"""
Generates corrupted variants of a spectrum CSV for robustness testing.
Produces three output files: noise-only, shift-only, and combined.

Usage:
    python scripts/corrupt_spectrum.py <spectrum.csv> [--noise 0.01] [--shift 0.05] [--seed 0]
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path


def load(filepath):
    wavenumbers, absorbance = [], []
    with open(filepath, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                if len(row) >= 2:
                    wavenumbers.append(float(row[0]))
                    absorbance.append(float(row[1]))
                else:
                    absorbance.append(float(row[0]))
            except ValueError:
                continue  # skip header
    wn = np.array(wavenumbers) if wavenumbers else None
    return wn, np.array(absorbance, dtype=np.float64)


def save(filepath, wavenumbers, absorbance):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        if wavenumbers is not None:
            writer.writerow(["wavenumber", "absorbance"])
            for wn, ab in zip(wavenumbers, absorbance):
                writer.writerow([wn, ab])
        else:
            writer.writerow(["absorbance"])
            for ab in absorbance:
                writer.writerow([ab])


def corrupt(input_path, noise_std=0.01, shift_amount=0.05, seed=0):
    rng = np.random.RandomState(seed)

    wavenumbers, spectrum = load(input_path)
    stem = Path(input_path).stem
    out_dir = Path(input_path).parent

    noise = rng.normal(0, noise_std, len(spectrum))
    # Random sign so the shift can go either up or down
    shift = rng.choice([-1, 1]) * shift_amount

    variants = {
        f"{stem}_noisy.csv":     (spectrum + noise,         f"Gaussian noise  std={noise_std}"),
        f"{stem}_shifted.csv":   (spectrum + shift,         f"vertical shift  {shift:+.4f}"),
        f"{stem}_corrupted.csv": (spectrum + noise + shift, f"noise + shift"),
    }

    for filename, (corrupted, desc) in variants.items():
        path = out_dir / filename
        save(path, wavenumbers, corrupted)
        print(f"  {filename:<35} {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spectrum", help="Input spectrum CSV file")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Standard deviation of Gaussian noise (default: 0.01)")
    parser.add_argument("--shift", type=float, default=0.05,
                        help="Magnitude of vertical shift (default: 0.05)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    args = parser.parse_args()

    print(f"Corrupting {args.spectrum}:")
    corrupt(args.spectrum, noise_std=args.noise, shift_amount=args.shift, seed=args.seed)
