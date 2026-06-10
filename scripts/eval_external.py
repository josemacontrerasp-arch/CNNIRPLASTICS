"""
Evaluate the saved 4-fold CNN ensemble on two EXTERNAL datasets the model has
never seen: FLOPP-e and BLoP.

Why this script exists
----------------------
The team wants to know how the classifier generalizes to spectra from other
instruments / labs (FLOPP-e) and to bioplastics that are NOT among the six
trained classes (BLoP).  This is a pure generalization test: nothing here is
trained, we only run inference with the existing output/cnn_fold_*.keras models.

Pipeline
--------
1. parse (wavenumber, %T) pairs from each external CSV
2. drop exact-zero %T points for BLoP (no-data marker; pre-interpolation
   data cleaning so they don't corrupt baseline correction)
3. interpolate onto the shared training grid
4. apply the pipeline PREPROCESS_CONFIG (same transforms used during training)
5. soft-vote across all saved cnn_fold_*.keras models (average softmax, argmax)

Important domain notes
----------------------
* Training classes (label map): HDPE LDPE PP PS PVC PET.
* FLOPP-e "PE" pellets are polyethylene but FLOPP-e does not distinguish HDPE
  vs LDPE -> a PE spectrum is counted CORRECT if predicted HDPE *or* LDPE.
* FLOPP-e spectra typically start at ~650 cm-1, so the 400..650 fingerprint
  region (rich in discriminative bands) is extrapolated flat. This is a real
  limitation of the source data, reported in the output.
* BLoP CSVs carry a no-data region at low wavenumbers stored as exact 0 %T.
  Those points are dropped before interpolation so they do not corrupt
  baseline correction (a data-cleaning step, not part of preprocessing).
* Every BLoP material and several FLOPP-e materials (ABS, EVA, EVOH, Nylon,
  PMMA, PLA, PHB, ...) are OUT-OF-DISTRIBUTION: the 6-class softmax has no
  "unknown" option, so we report what it guesses and how confident it is.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import preprocess
from run_pipeline import PREPROCESS_CONFIG


# --- Keras compat shim (identical to predict.py) --------------------------
_UNKNOWN_KWARGS = ("quantization_config",)
for _cls in (keras.layers.Dense, keras.layers.Conv1D, keras.layers.Conv2D,
             keras.layers.MaxPool1D, keras.layers.Flatten, keras.layers.Dropout,
             keras.layers.InputLayer):
    _orig = _cls.__init__
    def _make(_orig=_orig):
        def _patched(self, *args, **kwargs):
            for k in _UNKNOWN_KWARGS:
                kwargs.pop(k, None)
            return _orig(self, *args, **kwargs)
        return _patched
    _cls.__init__ = _make()
# --------------------------------------------------------------------------
INT_TO_LABEL = {0: "HDPE", 1: "LDPE", 2: "PP", 3: "PS", 4: "PVC", 5: "PET"}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}
C8_CSV = ROOT / "data" / "FTIR_PLASTIC_c8.csv"
MODELS_DIR = ROOT / "output"
OUT_DIR = ROOT / "results" / "external"
EXPECTED_LEN = 1868

# A PE spectrum is correct if the model says either HDPE or LDPE.
PE_ACCEPT = {"HDPE", "LDPE"}
# Canonical materials that the 6-class model is actually able to get right.
IN_DIST = {"PE", "PP", "PS", "PVC", "PET"}


# ---------------------------------------------------------------- grid / model
def get_training_wavenumber_grid():
    with open(C8_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        row = next(reader)
    payload = row[6:len(row) - 2]
    wn = np.array([float(payload[j]) for j in range(0, len(payload), 2)])
    assert len(wn) == EXPECTED_LEN, f"expected {EXPECTED_LEN} wavenumbers, got {len(wn)}"
    return wn


def load_models():
    paths = sorted(MODELS_DIR.glob("cnn_fold_*.keras"))
    if not paths:
        sys.exit(f"No fold_*.keras models in {MODELS_DIR}")
    models = [tf.keras.models.load_model(p) for p in paths]
    print(f"Loaded {len(models)} CNN fold models: {[p.name for p in paths]}")
    return models


# ---------------------------------------------------------------- CSV loaders
def _parse_xy_csv(path: Path):
    """Parse a 2-column numeric CSV of (wavenumber, %T).

    Tolerant to a leading metadata line and a 'cm-1,%T' header (FLOPP-e) and to
    scientific-notation values (BLoP).  Any line whose first two comma- or
    whitespace-separated tokens are not both floats is skipped.
    """
    xs, ys = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0]); y = float(parts[1])
            except ValueError:
                continue
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def load_and_align(path: Path, grid: np.ndarray, drop_zero=False):
    """Parse, clean, and interpolate one external CSV onto `grid`.

    drop_zero: BLoP CSVs use exact-0 %T to mark no-data regions at low
    wavenumbers.  Dropping them before interpolation prevents those zeroes
    from being treated as real signal and from corrupting baseline correction.
    This is a data-cleaning step; it happens before preprocessing.

    Returns (x, (wn_lo, wn_hi)) where x has shape (1, len(grid), 1),
    preprocessed identically to the training data via PREPROCESS_CONFIG.

    NOTE: `grid` must match the input length the saved models expect.
    Without OpenSpecy that is 1868 pts (c8 native); with OpenSpecy it is
    ~395 pts.  EXPECTED_LEN at the top of this file reflects this.
    """
    wn, ity = _parse_xy_csv(path)
    if drop_zero:
        keep = ity != 0.0
        wn, ity = wn[keep], ity[keep]
    order = np.argsort(wn)
    wn, ity = wn[order], ity[order]
    resampled = np.interp(grid, wn, ity)
    if resampled.max() == resampled.min():
        raise ValueError(f"flat spectrum in {path.name}")
    processed = preprocess.preprocess(resampled[np.newaxis, :], PREPROCESS_CONFIG)
    return processed[..., np.newaxis].astype(np.float32), (wn.min(), wn.max())


# ---------------------------------------------------------------- true labels
def floppe_material(fname: str) -> str:
    """FLOPP-e filename 'PET-1.csv' / 'Nylon-66.csv' -> material token."""
    stem = Path(fname).stem
    return re.split(r"[-_ ]", stem)[0] if not stem.lower().startswith("nylon") else "Nylon"


def blop_material(fname: str) -> str:
    """BLoP filename 'PLA Bioplastic 10. ...' -> material before 'Bioplastic'."""
    stem = Path(fname).stem
    return stem.split("Bioplastic")[0].strip()


# ---------------------------------------------------------------- inference
def predict_one(models, x):
    probs = np.mean([m.predict(x, verbose=0)[0] for m in models], axis=0)
    order = np.argsort(probs)[::-1]
    return order[0], probs, order[1]


def run_dataset(name, folder, glob_pat, material_fn, models, grid, drop_zero):
    rows = []
    files = sorted(folder.glob(glob_pat))
    print(f"\n=== {name}: {len(files)} spectra ===")
    for path in files:
        try:
            x, (lo, hi) = load_and_align(path, grid, drop_zero=drop_zero)
        except Exception as e:
            print(f"  {path.name:<45} ERROR: {e}")
            continue
        top, probs, second = predict_one(models, x)
        mat = material_fn(path.name)
        pred = INT_TO_LABEL[top]
        # Scoring: only in-distribution materials are gradeable.
        if mat.upper() in IN_DIST:
            truth = mat.upper()
            if truth == "PE":
                correct = pred in PE_ACCEPT
            else:
                correct = pred == truth
        else:
            truth, correct = "(OOD)", None
        rows.append({
            "dataset": name, "file": path.name, "material": mat,
            "true": truth, "pred": pred, "conf": float(probs[top]),
            "second": INT_TO_LABEL[second], "second_conf": float(probs[second]),
            "correct": correct, "wn_lo": lo, "wn_hi": hi,
            "probs": probs,
        })
        flag = "" if correct is None else ("OK " if correct else "XX ")
        print(f"  {flag}{path.name:<45} {mat:<12} -> {pred:<5} {probs[top]*100:5.1f}%"
              f"  (2nd {INT_TO_LABEL[second]} {probs[second]*100:.1f}%)")
    return rows


# ---------------------------------------------------------------- reporting
def write_report(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV (drop the bulky prob vector)
    csv_path = OUT_DIR / "external_predictions.csv"
    fields = ["dataset", "file", "material", "true", "pred", "conf",
              "second", "second_conf", "correct", "wn_lo", "wn_hi"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # In-distribution accuracy (FLOPP-e gradeable rows)
    graded = [r for r in rows if r["correct"] is not None]
    n_correct = sum(r["correct"] for r in graded)
    id_acc = n_correct / len(graded) if graded else float("nan")

    # Per-material breakdown
    mats = {}
    for r in rows:
        mats.setdefault(r["material"], []).append(r)

    md = []
    md.append("# External-Dataset Generalization Test\n")
    md.append("Soft-vote ensemble of `output/cnn_fold_1.keras` ... `cnn_fold_4.keras` "
              "(average of the 4 softmax outputs, then argmax).\n")
    md.append("Label map: `HDPE=0 LDPE=1 PP=2 PS=3 PVC=4 PET=5`. "
              "The model has **no \"unknown\" class** -- every spectrum is forced "
              "into one of these six.\n")
    md.append("## Headline\n")
    md.append(f"* **In-distribution accuracy (FLOPP-e):** {n_correct}/{len(graded)} "
              f"= **{id_acc*100:.1f}%** "
              f"(materials PE, PP, PS, PVC, PET; a PE spectrum counts correct if "
              f"predicted HDPE or LDPE).\n")
    ood = [r for r in rows if r["correct"] is None]
    md.append(f"* **Out-of-distribution spectra:** {len(ood)} "
              f"(bioplastics + polymers outside the six classes). These cannot be "
              f"\"correct\"; we report the forced guess and its confidence to show "
              f"whether the model over-confidently mislabels unknown polymers.\n")
    high_conf_ood = [r for r in ood if r["conf"] >= 0.9]
    md.append(f"* **{len(high_conf_ood)}/{len(ood)} OOD spectra were misclassified "
              f"with >=90% confidence** -- the softmax does not flag novelty, which "
              f"motivates adding an explicit reject option / the new bioplastic "
              f"classes.\n")

    # Per-material table
    md.append("\n## Per-material summary\n")
    md.append("| Material | n | In-dist? | Most common pred | Mean conf | Acc |")
    md.append("|---|---:|:--:|---|---:|---:|")
    for mat in sorted(mats):
        rs = mats[mat]
        preds = [r["pred"] for r in rs]
        common = max(set(preds), key=preds.count)
        mean_conf = np.mean([r["conf"] for r in rs])
        gradeable = [r for r in rs if r["correct"] is not None]
        acc = (f"{sum(x['correct'] for x in gradeable)}/{len(gradeable)}"
               if gradeable else "-")
        indist = "yes" if rs[0]["correct"] is not None else "OOD"
        md.append(f"| {mat} | {len(rs)} | {indist} | {common} | "
                  f"{mean_conf*100:.0f}% | {acc} |")

    # Full predictions table
    md.append("\n## All predictions\n")
    md.append("| Dataset | File | Material | True | Pred | Conf | 2nd guess |")
    md.append("|---|---|---|---|---|---:|---|")
    for r in sorted(rows, key=lambda r: (r["dataset"], r["material"], r["file"])):
        mark = {True: " ✓", False: " ✗", None: ""}[r["correct"]]
        md.append(f"| {r['dataset']} | {r['file']} | {r['material']} | {r['true']} | "
                  f"{r['pred']}{mark} | {r['conf']*100:.1f}% | "
                  f"{r['second']} ({r['second_conf']*100:.1f}%) |")

    md_path = OUT_DIR / "external_eval.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {md_path}")
    print(f"Wrote {csv_path}")

    # Confusion-style matrix for the gradeable (in-distribution) FLOPP-e rows:
    # rows = true material, cols = predicted class.
    _plot_id_confusion(graded)

    print(f"\nIn-distribution accuracy (FLOPP-e): {id_acc*100:.1f}% "
          f"({n_correct}/{len(graded)})")
    print(f"OOD spectra mislabeled with >=90% conf: "
          f"{len(high_conf_ood)}/{len(ood)}")


def _plot_id_confusion(graded):
    if not graded:
        return
    true_mats = ["PE", "PP", "PS", "PVC", "PET"]
    pred_cls = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]
    M = np.zeros((len(true_mats), len(pred_cls)), dtype=int)
    for r in graded:
        ti = true_mats.index(r["true"])
        pj = pred_cls.index(r["pred"])
        M[ti, pj] += 1
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(pred_cls)), pred_cls, rotation=45, ha="right")
    ax.set_yticks(range(len(true_mats)), true_mats)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True material (FLOPP-e)")
    ax.set_title("FLOPP-e in-distribution: true material vs predicted class")
    thr = M.max() / 2 if M.max() else 0.5
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center",
                        color="white" if M[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = OUT_DIR / "floppe_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    grid = get_training_wavenumber_grid()
    models = load_models()
    rows = []
    rows += run_dataset("FLOPP-e", ROOT / "data" / "external" / "flopp_e",
                        "*.csv", floppe_material, models, grid, drop_zero=False)
    rows += run_dataset("BLoP", ROOT / "data" / "external" / "blop",
                        "*.CSV", blop_material, models, grid, drop_zero=True)
    write_report(rows)


if __name__ == "__main__":
    main()
