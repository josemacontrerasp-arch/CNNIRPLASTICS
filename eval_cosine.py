"""
Evaluate the cosine library-match classifier -- the no-training floor.

Two regimes, plus a small preprocessing ablation:

(A) WITHIN-DATASET   5-fold StratifiedKFold per dataset (c4, c8, OpenSpecy).
(B) CROSS-DATASET    train templates on one dataset, test on another:
                     c4->c8, c8->c4, c4->OpenSpecy, c8->OpenSpecy.
(C) ABLATION         repeat both regimes for: raw, snv, asls, asls+snv.

All three datasets are placed on Jesse's shared 600-point wavenumber axis by
PlasticIRDataset, so spectra are directly comparable. Preprocessing is
per-spectrum and leakage-free, so it is applied once to each dataset outside the
CV loop; the only train-fold-only step is building the cosine templates.

Group IDs: each FTIR scan is a distinct physical sample (unique IDE; Sample runs
1..500 with no repeats), so StratifiedKFold introduces no replicate-scan leakage.
OpenSpecy exposes no particle ID. A warning is printed to flag this.

Outputs:
    results/cosine_eval.csv   summary table (rows=variant, cols=metrics)
    results/cosine_eval.md    readable version of the same table
    results/cm_*.png          confusion matrices

Usage:
    py eval_cosine.py
    py eval_cosine.py --cache output/_cache_processed.npz   # fast reruns
    py eval_cosine.py --samples path/to/txt_folder          # classify samples
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedKFold

# Make the loader's cm-1 prints survive a cp1252 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.append(str(Path(__file__).parent))
from data.format_data import PlasticIRDataset, POLYMER_CLASSES
from preprocess import preprocess, PreprocessConfig
from cosine_baseline import CosineLibraryClassifier

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
SOURCES = ["ftir_c4", "ftir_c8", "openspecy"]
SHORT = {"ftir_c4": "c4", "ftir_c8": "c8", "openspecy": "OpenSpecy"}

# Preprocessing variants for the ablation.
VARIANTS = {
    "raw":      PreprocessConfig(),
    "snv":      PreprocessConfig(normalize="snv"),
    "asls":     PreprocessConfig(baseline="asls"),
    "asls+snv": PreprocessConfig(baseline="asls", normalize="snv"),
}

# Cross-dataset permutations that we report.
CROSS_PAIRS = [
    ("ftir_c4", "ftir_c8"),
    ("ftir_c8", "ftir_c4"),
    ("ftir_c4", "openspecy"),
    ("ftir_c8", "openspecy"),
]

N_FOLDS = 5
RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_by_source(cache: str | None = None):
    """Return ({source: (X, y)}, wavenumbers) on the shared 600-pt axis."""
    if cache and Path(cache).exists():
        print(f"Loading cached arrays from {cache} ...")
        z = np.load(cache, allow_pickle=True)
        Xall, yall, srcall, wn = z["X"], z["y"], z["src"], z["wn"]
    else:
        ds = PlasticIRDataset(
            ftir_c4_path=str(DATA_DIR / "FTIR_PLASTIC_c4.csv"),
            ftir_c8_path=str(DATA_DIR / "FTIR_PLASTIC_c8.csv"),
            openspecy_dataset_path=str(DATA_DIR / "openspecy_polymer_dataset.csv"),
            openspecy_metadata_path=str(DATA_DIR / "openspecy_polymer_metadata.csv"),
            openspecy_wavenumbers_path=str(DATA_DIR / "openspecy_wavenumbers.csv"),
        )
        ds.process()
        data, wn = ds.get_formatted_data()
        Xall = np.array([d["intensities"] for d in data], dtype=float)
        yall = np.array([d["label"] for d in data])
        srcall = np.array([d["source"] for d in data])

    by = {}
    for s in SOURCES:
        m = srcall == s
        by[s] = (Xall[m], yall[m])
    return by, np.asarray(wn, dtype=float)


def preprocess_all(by_source, wn):
    """Apply each variant to each source once (leakage-free, outside CV).

    AsLS baseline correction is the only expensive step; it is computed once per
    (source) and shared between the 'asls' and 'asls+snv' variants.
    """
    from preprocess import snv, baseline_correct
    out = {v: {} for v in VARIANTS}
    for s in SOURCES:
        X, _ = by_source[s]
        X0 = np.nan_to_num(X, nan=0.0)
        print(f"  preprocessing {SHORT[s]} ({X0.shape[0]} spectra) ...")
        out["raw"][s] = X0
        out["snv"][s] = snv(X0)
        corrected = baseline_correct(X0, method="asls")
        out["asls"][s] = corrected
        out["asls+snv"][s] = snv(corrected)
    return out


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def metric_block(y_true, y_pred):
    return {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "macro_f1": skm.f1_score(y_true, y_pred, average="macro",
                                 labels=POLYMER_CLASSES, zero_division=0),
        "per_class_f1": dict(zip(
            POLYMER_CLASSES,
            skm.f1_score(y_true, y_pred, average=None,
                         labels=POLYMER_CLASSES, zero_division=0),
        )),
    }


def save_confusion(y_true, y_pred, title, path):
    cm = skm.confusion_matrix(y_true, y_pred, labels=POLYMER_CLASSES)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(POLYMER_CLASSES)), POLYMER_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(POLYMER_CLASSES)), POLYMER_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

def run_within(Xs, ys, variant, source):
    """5-fold StratifiedKFold cosine matching within one dataset."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_true_all, y_pred_all, fold_macro = [], [], []
    for tr, te in skf.split(Xs, ys):
        clf = CosineLibraryClassifier("centroid").fit(Xs[tr], ys[tr])
        yp = clf.predict(Xs[te])
        y_true_all.append(ys[te]); y_pred_all.append(yp)
        fold_macro.append(skm.f1_score(ys[te], yp, average="macro",
                                       labels=POLYMER_CLASSES, zero_division=0))
    y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all)
    block = metric_block(y_true, y_pred)
    block["macro_f1_fold_std"] = float(np.std(fold_macro))
    save_confusion(y_true, y_pred,
                   f"within {SHORT[source]} | {variant} | macroF1={block['macro_f1']:.3f}",
                   RESULTS_DIR / f"cm_within_{variant.replace('+','_')}_{SHORT[source]}.png")
    return block


def run_cross(X_train, y_train, X_test, y_test, variant, pair):
    """Train templates on one dataset, test on another (shared 6-class space)."""
    src, dst = pair
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    shared = sorted(train_classes & test_classes)
    dropped = sorted((train_classes | test_classes) - set(shared))

    m_tr = np.isin(y_train, shared)
    m_te = np.isin(y_test, shared)
    clf = CosineLibraryClassifier("centroid").fit(X_train[m_tr], y_train[m_tr])
    y_pred = clf.predict(X_test[m_te])
    y_true = y_test[m_te]

    block = metric_block(y_true, y_pred)
    block["dropped_classes"] = dropped
    save_confusion(y_true, y_pred,
                   f"{SHORT[src]}->{SHORT[dst]} | {variant} | macroF1={block['macro_f1']:.3f}",
                   RESULTS_DIR / f"cm_cross_{variant.replace('+','_')}_{SHORT[src]}_to_{SHORT[dst]}.png")
    return block


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_table(within, cross):
    """Rows = variant; columns = within macro-F1 per dataset + each cross pair."""
    within_cols = [f"within_{SHORT[s]}_macroF1" for s in SOURCES]
    cross_cols = [f"{SHORT[a]}->{SHORT[b]}_macroF1" for a, b in CROSS_PAIRS]
    header = ["variant"] + within_cols + cross_cols
    rows = []
    for v in VARIANTS:
        row = [v]
        row += [f"{within[v][s]['macro_f1']:.3f}" for s in SOURCES]
        row += [f"{cross[v][(a, b)]['macro_f1']:.3f}" for a, b in CROSS_PAIRS]
        rows.append(row)
    return header, rows


def write_outputs(header, rows, within, cross):
    RESULTS_DIR.mkdir(exist_ok=True)
    import csv
    with open(RESULTS_DIR / "cosine_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    lines = ["# Cosine library-match baseline -- evaluation", "",
             "Nearest-reference (centroid) classifier under cosine similarity.",
             "This is the no-training floor any ML model must beat.", "",
             "## Summary: macro-F1", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    lines += ["", "## Within-dataset detail", ""]
    for v in VARIANTS:
        lines.append(f"### {v}")
        for s in SOURCES:
            b = within[v][s]
            pc = ", ".join(f"{k}={x:.2f}" for k, x in b["per_class_f1"].items())
            lines.append(f"- **{SHORT[s]}**: acc={b['accuracy']:.3f}, "
                         f"macroF1={b['macro_f1']:.3f} (fold std {b['macro_f1_fold_std']:.3f}); "
                         f"per-class F1: {pc}")
        lines.append("")

    lines += ["## Cross-dataset detail", ""]
    for v in VARIANTS:
        lines.append(f"### {v}")
        for a, b_ in CROSS_PAIRS:
            b = cross[v][(a, b_)]
            drop = b["dropped_classes"] or "none"
            pc = ", ".join(f"{k}={x:.2f}" for k, x in b["per_class_f1"].items())
            lines.append(f"- **{SHORT[a]}->{SHORT[b_]}**: acc={b['accuracy']:.3f}, "
                         f"macroF1={b['macro_f1']:.3f} (dropped classes: {drop}); "
                         f"per-class F1: {pc}")
        lines.append("")

    with open(RESULTS_DIR / "cosine_eval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_table(header, rows):
    widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print("\n" + fmt.format(*header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


# ---------------------------------------------------------------------------
# Optional --samples hook
# ---------------------------------------------------------------------------

def classify_samples(folder, by_source, wn):
    """Classify our own measured .txt spectra against a c4+c8 library."""
    from predict import load_txt_spectrum
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"No .txt files in {folder}"); return

    Xc4, yc4 = by_source["ftir_c4"]; Xc8, yc8 = by_source["ftir_c8"]
    X_lib = np.nan_to_num(np.vstack([Xc4, Xc8]), nan=0.0)
    y_lib = np.concatenate([yc4, yc8])
    from preprocess import snv
    clf = CosineLibraryClassifier("centroid").fit(snv(X_lib), y_lib)

    print(f"\n{'file':<28} {'pred':<6} {'cosine':>7}")
    print("-" * 45)
    for p in files:
        x, inten = load_txt_spectrum(p)
        order = np.argsort(x)
        resampled = np.interp(wn, x[order], inten[order], left=np.nan, right=np.nan)
        Xs = snv(np.nan_to_num(resampled[None, :], nan=0.0))
        lbl, score = clf.predict_with_score(Xs)
        print(f"{p.name:<28} {lbl[0]:<6} {score[0]:7.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Cosine library-match evaluation")
    ap.add_argument("--cache", default=None, help="path to a processed .npz for fast reruns")
    ap.add_argument("--samples", default=None, help="folder of .txt spectra to classify")
    args = ap.parse_args()

    print("Loading data on the shared wavenumber axis ...")
    by_source, wn = load_by_source(args.cache)
    for s in SOURCES:
        print(f"  {SHORT[s]:<10} X={by_source[s][0].shape}  classes={sorted(set(by_source[s][1]))}")

    if args.samples:
        classify_samples(args.samples, by_source, wn)
        return

    print("\nWARNING: OpenSpecy exposes no particle/sample ID; within-OpenSpecy "
          "StratifiedKFold results may be optimistic if replicate scans exist.")

    print("\nApplying preprocessing variants (leakage-free, once per dataset) ...")
    pp = preprocess_all(by_source, wn)
    labels = {s: by_source[s][1] for s in SOURCES}

    within = {v: {} for v in VARIANTS}
    cross = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        print(f"\n=== variant: {v} ===")
        for s in SOURCES:
            within[v][s] = run_within(pp[v][s], labels[s], v, s)
            print(f"  within {SHORT[s]:<10} macroF1={within[v][s]['macro_f1']:.3f} "
                  f"acc={within[v][s]['accuracy']:.3f}")
        for a, b in CROSS_PAIRS:
            cross[v][(a, b)] = run_cross(pp[v][a], labels[a], pp[v][b], labels[b], v, (a, b))
            print(f"  cross  {SHORT[a]}->{SHORT[b]:<10} macroF1={cross[v][(a, b)]['macro_f1']:.3f} "
                  f"acc={cross[v][(a, b)]['accuracy']:.3f}")

    header, rows = build_table(within, cross)
    write_outputs(header, rows, within, cross)
    print_table(header, rows)
    print(f"\nSaved results/cosine_eval.csv, results/cosine_eval.md, and confusion matrices to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
