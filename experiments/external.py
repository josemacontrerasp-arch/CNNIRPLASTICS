"""Loaders + scoring for held-out spectra that never enter training:

  * lab     -- the 39 real-world FTIR samples (data/ftir_real_world_samples/*.txt)
  * FLOPP-e -- weathered-plastic FTIR library (data/external/flopp_e/*.csv)
  * BLoP    -- bioplastics FTIR library       (data/external/blop/*.CSV)

All three are aligned to a target grid with EDGE-CLAMPING (np.interp), matching
the convention in scripts/eval_external.py. Clamping (rather than the strict
NaN-drop used for the training grid) is what lets the lab spectra survive on the
fine 399-4000 grid -- they overhang the grid endpoints by ~2 cm^-1 and would
otherwise all be dropped. The clamped edge region is flat and tiny; for FLOPP-e
(which starts ~650 cm^-1) the 400-650 region is extrapolated flat, a real source
limitation we report rather than hide.

Preprocessing is NOT applied here. The caller applies the SAME PreprocessConfig
used for training, so external spectra land on identical columns.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from data.format_data import LAB_LABELS
from experiments import config as C


# --------------------------------------------------------------------------- #
# Parsing / alignment
# --------------------------------------------------------------------------- #
def parse_xy(path: Path):
    """Parse a 2-column numeric file of (wavenumber, intensity/%T).

    Tolerant to header/metadata lines and scientific notation; any line whose
    first two comma/space-separated tokens are not both floats is skipped.
    """
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, float)


def align_clamp(wn, ity, grid, drop_zero=False):
    """Resample one spectrum onto `grid` by linear interp, clamping at edges.

    drop_zero: BLoP marks no-data with exact-0 %T at low wavenumbers; dropping
    those before interpolation prevents the zeros being read as real signal.
    Returns the RAW aligned vector (no normalization). Raises on a flat spectrum.
    """
    if drop_zero:
        keep = ity != 0.0
        wn, ity = wn[keep], ity[keep]
    order = np.argsort(wn)
    wn, ity = wn[order], ity[order]
    res = np.interp(grid, wn, ity)            # np.interp clamps out-of-range to edges
    if res.max() == res.min():
        raise ValueError("flat spectrum")
    return res.astype(np.float32)


# --------------------------------------------------------------------------- #
# Material -> truth helpers
# --------------------------------------------------------------------------- #
def floppe_material(fname: str) -> str:
    stem = Path(fname).stem
    return "Nylon" if stem.lower().startswith("nylon") else re.split(r"[-_ ]", stem)[0]


def blop_material(fname: str) -> str:
    return Path(fname).stem.split("Bioplastic")[0].strip()


# --------------------------------------------------------------------------- #
# Dataset loaders -> list of row dicts with a raw aligned vector 'x'
# --------------------------------------------------------------------------- #
def load_lab(grid):
    rows = []
    for path in sorted(C.LAB_FOLDER.glob("*.txt")):
        stem = path.stem
        if stem not in LAB_LABELS:
            continue
        wn, ity = parse_xy(path)
        try:
            x = align_clamp(wn, ity, grid, drop_zero=False)
        except ValueError:
            continue
        truth = LAB_LABELS[stem]            # one of the six commodity classes
        rows.append(dict(dataset="lab", file=path.name, material=truth,
                         truth=truth, x=x, ood=False))
    return rows


def load_floppe(grid):
    rows = []
    for path in sorted(C.FLOPPE_DIR.glob("*.csv")):
        wn, ity = parse_xy(path)
        try:
            x = align_clamp(wn, ity, grid, drop_zero=False)
        except ValueError:
            continue
        mat = floppe_material(path.name)
        # FLOPP-e %T -> the model is trained on absorbance-like values; FLOPP-e is
        # %T so we keep it as-is and rely on per-spectrum normalization, matching
        # the existing eval_external convention.
        truth = mat.upper() if mat.upper() in C.IN_DIST_MATERIALS else None
        rows.append(dict(dataset="FLOPP-e", file=path.name, material=mat,
                         truth=truth, x=x, ood=(truth is None)))
    return rows


def load_blop(grid):
    rows = []
    for path in sorted(C.BLOP_DIR.glob("*.CSV")):
        wn, ity = parse_xy(path)
        try:
            x = align_clamp(wn, ity, grid, drop_zero=True)
        except ValueError:
            continue
        mat = blop_material(path.name)
        rows.append(dict(dataset="BLoP", file=path.name, material=mat,
                         truth=None, x=x, ood=True))   # all bioplastics are OOD
    return rows


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_indist(rows, pred_labels, conf):
    """In-distribution accuracy over gradeable rows (truth is one of the six).

    FLOPP-e/lab 'PE' counts correct if predicted HDPE or LDPE. Returns dict with
    accuracy, per-row records, and an OOD high-confidence count.
    """
    n_id = n_ok = 0
    n_ood = n_ood_hi = 0
    records = []
    for r, p, c in zip(rows, pred_labels, conf):
        rec = dict(dataset=r["dataset"], file=r["file"], material=r["material"],
                   truth=r["truth"] or "OOD", pred=p, conf=float(c))
        records.append(rec)
        if r["truth"] is not None:
            n_id += 1
            ok = (p in C.PE_ACCEPT) if r["truth"] == "PE" else (p == r["truth"])
            n_ok += int(ok)
            rec["correct"] = bool(ok)
        else:
            n_ood += 1
            n_ood_hi += int(c >= 0.9)
            rec["correct"] = None
    return dict(id_acc=(n_ok / n_id if n_id else float("nan")),
                n_ok=n_ok, n_id=n_id, n_ood=n_ood, n_ood_hi=n_ood_hi,
                records=records)
