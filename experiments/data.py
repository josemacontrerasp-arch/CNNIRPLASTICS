"""Data assembly + caching.

The expensive step in every experiment is loading the ~400 MB of FTIR/OpenSpecy
CSVs and aligning them onto a shared wavenumber grid. We do that ONCE per
"arm" (with vs without OpenSpecy), cache the raw-aligned matrix + string labels
to .npy, and let the stage scripts apply cheap per-config preprocessing on top.

This module deliberately bypasses run_pipeline.extract_data because the
regenerated OpenSpecy now contains PBS/PBAT labels that are absent from the
8-class map; extract_data raises on those. Here we keep labels as strings and
filter by scope ourselves, so non-target spectra (incl. bioplastics) are simply
dropped from the six-class task or pooled into OTHER -- never a crash.
"""
from __future__ import annotations

import numpy as np

from data.format_data import PlasticIRDataset
from experiments import config as C


# --------------------------------------------------------------------------- #
# Raw-aligned assembly (cached)
# --------------------------------------------------------------------------- #
def _cache_paths(include_openspecy: bool):
    tag = "with_os" if include_openspecy else "no_os"
    return {
        "X": C.CACHE_DIR / f"X_{tag}.npy",
        "labels": C.CACHE_DIR / f"labels_{tag}.npy",
        "sources": C.CACHE_DIR / f"sources_{tag}.npy",
        "wn": C.CACHE_DIR / f"wn_{tag}.npy",
        "Xlab": C.CACHE_DIR / f"Xlab_{tag}.npy",
        "lab_labels": C.CACHE_DIR / f"lab_labels_{tag}.npy",
    }


def assemble(include_openspecy: bool, use_cache: bool = True, include_lab: bool = True):
    """Load + align all training sources onto one grid; cache the result.

    Returns a dict:
        X        : (n, p) float32   raw-aligned absorbance (NO preprocessing yet)
        labels   : (n,) str         polymer label strings (incl. bioplastics)
        sources  : (n,) str         'ftir_c4' | 'ftir_c8' | 'openspecy'
        wn       : (p,) float64     shared wavenumber axis (cm^-1)
        X_lab    : (m, p) float32 | None   test-only lab spectra, same grid
        lab_labels: (m,) str | None
    Lab spectra are kept SEPARATE (test-only); they never influence the grid
    because they are aligned after process() decides it.
    """
    paths = _cache_paths(include_openspecy)
    if use_cache and paths["X"].exists():
        X = np.load(paths["X"])
        labels = np.load(paths["labels"], allow_pickle=True)
        sources = np.load(paths["sources"], allow_pickle=True)
        wn = np.load(paths["wn"])
        X_lab = np.load(paths["Xlab"], allow_pickle=True) if paths["Xlab"].exists() else None
        lab_labels = (np.load(paths["lab_labels"], allow_pickle=True)
                      if paths["lab_labels"].exists() else None)
        if X_lab is not None and X_lab.dtype == object:
            X_lab = None  # sentinel for "no lab"
        return dict(X=X, labels=labels, sources=sources, wn=wn,
                    X_lab=X_lab, lab_labels=lab_labels)

    ds = PlasticIRDataset(
        ftir_c4_path=str(C.DATA_DIR / "FTIR_PLASTIC_c4.csv"),
        ftir_c8_path=str(C.DATA_DIR / "FTIR_PLASTIC_c8.csv"),
        openspecy_dataset_path=str(C.DATA_DIR / "openspecy_polymer_dataset.csv"),
        openspecy_metadata_path=str(C.DATA_DIR / "openspecy_polymer_metadata.csv"),
        openspecy_wavenumbers_path=str(C.DATA_DIR / "openspecy_wavenumbers.csv"),
    )
    ds.load_raw()
    if not include_openspecy:
        # Dropping the key removes OpenSpecy from the grid decision AND the output
        # (load_raw is idempotent, so process() won't re-add it).
        ds._raw_data.pop("openspecy", None)
    ds.process()
    formatted, wn = ds.get_formatted_data()
    wn = np.asarray(wn, float)

    X = np.asarray([e["intensities"] for e in formatted], dtype=np.float32)
    labels = np.asarray([str(e["label"]) for e in formatted], dtype=object)
    sources = np.asarray([e["source"] for e in formatted], dtype=object)

    # Test-only lab data, aligned to the SAME grid (after process()).
    X_lab = lab_labels = None
    if include_lab and C.LAB_FOLDER.exists():
        n_train = len(formatted)
        ds.add_lab_data(str(C.LAB_FOLDER))
        lab = [e for e in ds.formatted_data[n_train:]
               if not np.isnan(e["intensities"]).any()]
        if lab:
            X_lab = np.asarray([e["intensities"] for e in lab], dtype=np.float32)
            lab_labels = np.asarray([str(e["label"]) for e in lab], dtype=object)

    np.save(paths["X"], X)
    np.save(paths["labels"], labels)
    np.save(paths["sources"], sources)
    np.save(paths["wn"], wn)
    if X_lab is not None:
        np.save(paths["Xlab"], X_lab)
        np.save(paths["lab_labels"], lab_labels)

    return dict(X=X, labels=labels, sources=sources, wn=wn,
                X_lab=X_lab, lab_labels=lab_labels)


# --------------------------------------------------------------------------- #
# Scope filtering / label mapping
# --------------------------------------------------------------------------- #
def build_xy(assembled: dict, scope: str = "six", drop_openspecy_pvc_dupe=False):
    """Filter assembled spectra to a class scope and map to integer labels.

    scope:
      "six"        -> only HDPE/LDPE/PP/PS/PVC/PET (bioplastics dropped).
      "six+other"  -> six commodity classes plus an OTHER class built from
                      bioplastic spectra (PLA/PHA/PBS/...). Tests the
                      "reject option as an explicit class" idea.

    Returns X (float32), y (int), label_names (list[str]).
    Also returns the OOD pool (bioplastic spectra) when scope == "six", so the
    caller can use them as a held-out novelty set if desired.
    """
    X, labels = assembled["X"], assembled["labels"]

    if scope == "six":
        mask = np.array([lab in C.SIX_TO_INT for lab in labels])
        Xs = X[mask]
        y = np.array([C.SIX_TO_INT[lab] for lab in labels[mask]], dtype=int)
        return Xs, y, list(C.SIX)

    if scope == "six+other":
        keep = np.array([(lab in C.SIX_TO_INT) or (lab in C.BIOPLASTIC_LABELS)
                         for lab in labels])
        Xs = X[keep]
        ys = []
        for lab in labels[keep]:
            if lab in C.SIX_TO_INT:
                ys.append(C.SIX_TO_INT[lab])
            else:
                ys.append(C.SIX_PLUS_OTHER_TO_INT["OTHER"])
        return Xs, np.array(ys, dtype=int), list(C.SIX_PLUS_OTHER)

    raise ValueError(f"unknown scope '{scope}'")


def bioplastic_pool(assembled: dict):
    """Return (X, labels) of the bioplastic spectra in the assembled set.

    Useful as an in-library novelty/OOD set distinct from the external BLoP
    files (these come from the same OpenSpecy instrument distribution).
    """
    X, labels = assembled["X"], assembled["labels"]
    mask = np.array([lab in C.BIOPLASTIC_LABELS for lab in labels])
    return X[mask], labels[mask]
