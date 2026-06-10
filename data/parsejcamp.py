"""
Adversarial probe: feed non-plastic reference spectra through the trained CNN
and measure how confidently it misclassifies them.

Molecules chosen because they share key bond signatures with specific plastic classes:
  - Stearic acid           → shares C-H stretches with HDPE/LDPE (long CH2 chain)
  - Ethylbenzene           → shares aromatic ring with PS
  - Ethyl acetate          → shares ester C=O and C-O-C with PET
  - Polyisobutylene        → shares polyolefin backbone with PP
  - Chlorinated paraffin   → shares C-Cl stretches with PVC (manually digitised)

NIST spectra fetched directly from NIST WebBook (no login required, JCAMP-DX format).
Chlorinated paraffin synthesised from peaks manually read off a reference spectrum image.
"""

import csv
import re
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d
from tensorflow import keras

# ── Configuration ---──────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

PROBE_MOLECULES = [
    {
        "name":        "Stearic acid",
        "target_class": "HDPE/LDPE",
        "reason":      "Long CH2 chain; shares C-H stretches with polyethylene",
        "jcamp_url":   "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C57114&Index=0&Type=IR",
    },
    {
        "name":        "Ethylbenzene",
        "target_class": "PS",
        "reason":      "Aromatic ring + alkyl C-H; main PS fingerprint",
        "jcamp_url":   "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C100414&Index=0&Type=IR",
    },
    {
        "name":        "Ethyl acetate",
        "target_class": "PET",
        "reason":      "Ester C=O (~1735 cm-1) and C-O-C stretch; PET's dominant peaks",
        "jcamp_url":   "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C141786&Index=0&Type=IR",
    },
    {
        "name":        "Polyisobutylene",
        "target_class": "PP",
        "reason":      "Polyolefin backbone; similar C-H bending and stretching to PP",
        "jcamp_url":   "https://webbook.nist.gov/cgi/cbook.cgi?JCAMP=C9003274&Index=0&Type=IR",
    },
]

CLASS_NAMES = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]

# Peaks manually digitised from a reference chlorinated paraffin FTIR spectrum.
# Format: (wavenumber cm-1, absorbance, half-width cm-1)
# C-Cl stretches (540-730 cm-1) are the PVC-like diagnostic region.
CHLORINATED_PARAFFIN_PEAKS = [
    (3500, 0.10, 120),   # broad O-H/N-H noise / moisture
    (2960, 0.28,  20),   # C-H asymmetric stretch
    (2870, 0.08,  15),   # C-H symmetric stretch (shoulder)
    (1465, 0.47,  18),   # CH2/CH3 scissor bend
    (1380, 0.15,  15),   # CH3 symmetric bend
    (1250, 0.25,  25),   # C-C-Cl wag
    (1150, 0.35,  30),   # C-C stretch / CH2 wag
    (1060, 0.65,  25),   # C-C-Cl stretch
    (1000, 0.30,  20),
    ( 960, 0.20,  18),
    ( 800, 0.35,  20),
    ( 730, 0.50,  15),   # C-Cl stretch
    ( 700, 0.55,  12),
    ( 680, 0.60,  12),
    ( 650, 0.70,  12),
    ( 620, 0.80,  10),   # strong C-Cl band
    ( 600, 0.82,  10),
    ( 570, 0.90,  10),
    ( 540, 1.00,  12),   # tallest peak — primary C-Cl stretch
    ( 510, 0.65,  15),
]

# ── Helpers ---─────────────────────────────────────────────────────────────────

def _get_training_wavenumbers() -> np.ndarray:
    """Extract the 1868-point wavenumber grid from the c8 CSV (first data row).
    Mirrors cnn_model_draft.py: row[6:len(row)-2] drops the trailing incomplete pair."""
    c8_path = BASE_DIR / "data" / "FTIR_PLASTIC_c8.csv"
    with open(c8_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)           # skip header
        row = next(reader)     # first data row
    # Match the exact slice used in cnn_model_draft.load_and_preprocess()
    data_cols = row[6:len(row) - 2]
    wavenumbers = np.array([float(data_cols[i]) for i in range(0, len(data_cols), 2)])
    assert len(wavenumbers) == 1868, f"Expected 1868 wavenumber points, got {len(wavenumbers)}"
    return wavenumbers


def _parse_jcamp(text: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a minimal JCAMP-DX IR spectrum into (wavenumbers, absorbances).
    Handles both XYDATA=(X++(Y..Y)) packed format and explicit XY pairs.
    """
    lines = text.splitlines()

    # Read header metadata
    firstx = lastx = xfactor = yfactor = deltax = None
    npoints = None
    data_start = None

    for i, line in enumerate(lines):
        line_upper = line.upper().strip()
        if line_upper.startswith("##FIRSTX="):
            firstx = float(line.split("=", 1)[1])
        elif line_upper.startswith("##LASTX="):
            lastx = float(line.split("=", 1)[1])
        elif line_upper.startswith("##XFACTOR="):
            xfactor = float(line.split("=", 1)[1])
        elif line_upper.startswith("##YFACTOR="):
            yfactor = float(line.split("=", 1)[1])
        elif line_upper.startswith("##DELTAX="):
            deltax = float(line.split("=", 1)[1])
        elif line_upper.startswith("##NPOINTS="):
            npoints = int(line.split("=", 1)[1])
        elif "##XYDATA" in line_upper or "##XYPOINTS" in line_upper:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("No XYDATA or XYPOINTS block found in JCAMP file.")

    xfactor = xfactor or 1.0
    yfactor = yfactor or 1.0

    # Collect raw data lines until END
    data_lines = []
    for line in lines[data_start:]:
        if line.strip().upper().startswith("##END"):
            break
        data_lines.append(line.strip())

    # Attempt to parse as packed X++(Y..Y) format first
    # In this format each line starts with an X value followed by Y values
    all_x, all_y = [], []

    for line in data_lines:
        if not line:
            continue
        # Split on whitespace or commas; numbers may be signed integers or floats
        tokens = re.split(r"[\s,]+", line.strip())
        if not tokens:
            continue
        try:
            nums = [float(t) for t in tokens if t]
        except ValueError:
            continue

        if len(nums) < 2:
            continue

        x_start = nums[0] * xfactor
        y_vals  = [v * yfactor for v in nums[1:]]

        # X increment per Y value in this line
        if deltax is not None:
            dx = deltax * xfactor
        elif npoints and firstx is not None and lastx is not None:
            dx = (lastx - firstx) / (npoints - 1) * xfactor / xfactor  # already scaled
            dx = (lastx - firstx) / (npoints - 1)
        else:
            dx = None

        if dx is not None:
            for j, y in enumerate(y_vals):
                all_x.append(x_start + j * dx)
                all_y.append(y)
        else:
            # Treat as (x, y) pairs on each line
            all_x.append(nums[0] * xfactor)
            all_y.append(nums[1] * yfactor)

    wavenumbers = np.array(all_x)
    absorbances = np.array(all_y)

    # Sort by wavenumber (NIST files are usually descending)
    order = np.argsort(wavenumbers)
    return wavenumbers[order], absorbances[order]


def _fetch_jcamp(url: str) -> tuple[np.ndarray, np.ndarray]:
    """Download a JCAMP-DX file from NIST and parse it."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return _parse_jcamp(text)


def _synthesise_from_peaks(
    peaks: list[tuple[float, float, float]],
    target_wn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a spectrum by summing Gaussian peaks, then return (wn, absorbance)."""
    absorbance = np.zeros_like(target_wn, dtype=float)
    for centre, height, hwhm in peaks:
        sigma = hwhm / (2 * np.sqrt(2 * np.log(2)))
        absorbance += height * np.exp(-0.5 * ((target_wn - centre) / sigma) ** 2)
    return target_wn.copy(), absorbance


def _preprocess(wn_src: np.ndarray, abs_src: np.ndarray,
                target_wn: np.ndarray) -> np.ndarray:
    """Interpolate to target grid and normalize to [0, 1]."""
    f = interp1d(wn_src, abs_src, kind="linear", bounds_error=False, fill_value=np.nan)
    aligned = f(target_wn)

    # Fill any out-of-range NaNs with 0
    aligned = np.where(np.isnan(aligned), 0.0, aligned)

    lo, hi = aligned.min(), aligned.max()
    if hi > lo:
        aligned = (aligned - lo) / (hi - lo)

    return aligned.astype(np.float32)


# ── Main ---────────────────────────────────────────────────────────────────────

def main():
    # Load wavenumber grid from training data
    print("Extracting training wavenumber grid from c8 CSV …")
    target_wn = _get_training_wavenumbers()
    print(f"  Grid: {target_wn[0]:.1f} – {target_wn[-1]:.1f} cm-1  ({len(target_wn)} points)\n")

    # Load saved models (use all available folds for an ensemble)
    fold_paths = sorted((BASE_DIR / "output").glob("fold_*.keras"))
    if not fold_paths:
        raise FileNotFoundError("No fold_*.keras models found in output/")
    print(f"Loading {len(fold_paths)} model(s): {[p.name for p in fold_paths]}\n")
    models = [keras.models.load_model(p) for p in fold_paths]

    # Warm up
    dummy = tf.zeros((1, 1868, 1), dtype=tf.float32)
    for m in models:
        m(dummy, training=False)

    # Run probe
    results = []
    for mol in PROBE_MOLECULES:
        print(f"---{mol['name']} (proxy for {mol['target_class']}) ---")
        try:
            wn, ab = _fetch_jcamp(mol["jcamp_url"])
            print(f"  Downloaded: {len(wn)} points, "
                  f"{wn.min():.0f}–{wn.max():.0f} cm-1")
        except Exception as e:
            print(f"  FAILED to fetch: {e}")
            results.append({**mol, "error": str(e)})
            continue

        spectrum = _preprocess(wn, ab, target_wn)
        x = tf.constant(spectrum[np.newaxis, :, np.newaxis])  # (1, 1868, 1)

        # Ensemble: average softmax outputs across folds
        probs_all = np.stack([m(x, training=False).numpy()[0] for m in models])
        probs = probs_all.mean(axis=0)

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        print(f"  Predicted:  {CLASS_NAMES[pred_idx]}  ({confidence*100:.1f}% confidence)")
        print(f"  All class probabilities:")
        for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
            marker = " <--" if i == pred_idx else ""
            print(f"    {cls:6s}  {p*100:5.1f}%{marker}")
        print()

        results.append({
            **mol,
            "predicted":    CLASS_NAMES[pred_idx],
            "confidence":   confidence,
            "probs":        probs.tolist(),
        })

    # ── Chlorinated paraffin (synthetic from digitised peaks) ---──────────────
    print("---Chlorinated paraffin (PVC proxy — manually digitised) ---")
    wn_synth, ab_synth = _synthesise_from_peaks(CHLORINATED_PARAFFIN_PEAKS, target_wn)
    spectrum = _preprocess(wn_synth, ab_synth, target_wn)
    x = tf.constant(spectrum[np.newaxis, :, np.newaxis])

    probs_all = np.stack([m(x, training=False).numpy()[0] for m in models])
    probs = probs_all.mean(axis=0)
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    print(f"  Predicted:  {CLASS_NAMES[pred_idx]}  ({confidence*100:.1f}% confidence)")
    print(f"  All class probabilities:")
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        marker = " <--" if i == pred_idx else ""
        print(f"    {cls:6s}  {p*100:5.1f}%{marker}")
    print()

    results.append({
        "name":         "Chlorinated paraffin",
        "target_class": "PVC",
        "reason":       "C-Cl stretches (540-730 cm-1); manually digitised from spectrum image",
        "predicted":    CLASS_NAMES[pred_idx],
        "confidence":   confidence,
        "probs":        probs.tolist(),
        "synthetic":    True,
    })

    # Summary table
    print("=" * 65)
    print(f"{'Molecule':<22} {'Proxy for':<10} {'Predicted':<8} {'Confidence':>10}")
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<22} {r['target_class']:<10} {'ERROR':<8} {'—':>10}")
        else:
            synth_marker = " *" if r.get("synthetic") else ""
            flag = " !" if r["confidence"] > 0.60 else ""
            print(f"{r['name']:<22} {r['target_class']:<10} {r['predicted']:<8} "
                  f"{r['confidence']*100:9.1f}%{flag}{synth_marker}")
    print("=" * 65)
    print("! = confidence > 60% (potential bond over-reliance)")
    print("* = synthesised from manually digitised peaks (not a raw JCAMP file)")


if __name__ == "__main__":
    main()