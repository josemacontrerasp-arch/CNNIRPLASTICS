"""
Test every trained model family on the new real-world spectra in
C:/Users/Josem/Downloads/data and produce per-sample saliency maps.

Model families covered (all that exist in the repo as of 2026-06-12):

  legacy-cnn   output/fold_1..4.keras            1868-pt grid, smooth+asls+snv
  legacy-rf    output/rf_fold_1..5.pkl            600-pt grid, smooth+asls+snv
  final-rf     experiments_output/models/final/rf_final.pkl (+5 fold soft-vote)
                                                 1868-pt grid, snv, reject@0.89
  final-cnn    experiments_output/models/final/cnn_fold_1..5.keras
                                                 1868-pt grid, snv  (deployment CNN)
  cnn-d1       experiments_output/models/no_os__smooth+d1+snv/*.keras
                                                 1868-pt grid, smooth+d1+snv
  cnn-os       experiments_output/models/with_os__norm-snv/*.keras
                                                 395-pt grid (804-3168), snv

Saliency (gradient + 1-D Grad-CAM, ensembled over the 5 final-CNN folds) is
computed per sample for the deployment CNN's predicted class.

Outputs -> reports/lab_test_2026-06-12/
  predictions.csv         every model family x every file
  predictions.md          human-readable summary table
  saliency/<file>.png     per-sample saliency overlay
  saliency_overview.png   all samples stacked (Grad-CAM)
"""
import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import keras
import tensorflow as tf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Keras compat shim (models saved on Colab with a newer Keras) ----------
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
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from preprocess import preprocess, PreprocessConfig  # noqa: E402

DATA_DIR = Path(r"C:\Users\Josem\Downloads\data")
OUT_DIR = ROOT / "reports" / "lab_test_2026-06-12"
SAL_DIR = OUT_DIR / "saliency"
EXP_MODELS = ROOT / "experiments_output" / "models"
CACHE = ROOT / "experiments_output" / "cache"

SIX = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]
REJECT_THRESHOLD = 0.89          # deployment.json reject rule (final-rf only)

CFG_SNV = PreprocessConfig(normalize="snv")
CFG_LEGACY = PreprocessConfig(smooth=True, baseline="asls", normalize="snv")
CFG_D1 = PreprocessConfig(smooth=True, derivative=1, normalize="snv")


# ---------------------------------------------------------------- loading
def parse_xy(path):
    """Two-column (wavenumber, intensity) with ##-style header lines."""
    xs, ys = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("##"):
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


def align_clamp(wn, ity, grid):
    """Linear interp onto grid; out-of-range clamps to edge values."""
    order = np.argsort(wn)
    return np.interp(grid, wn[order], ity[order])


def load_keras_ensemble(paths):
    return [keras.models.load_model(p) for p in paths]


def load_rf_ensemble(paths):
    return [pickle.load(open(p, "rb")) for p in paths]


def rf_proba(forests, X):
    acc = np.zeros((len(X), len(SIX)))
    for rf in forests:
        p = np.zeros((len(X), len(SIX)))
        p[:, rf.classes_] = rf.predict_proba(X)
        acc += p
    return acc / len(forests)


def cnn_proba(models, X):
    X = np.asarray(X, np.float32)[..., np.newaxis]
    return np.mean([m.predict(X, verbose=0) for m in models], axis=0)


# ---------------------------------------------------------------- saliency
def last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv1D):
            return layer
    raise RuntimeError("no Conv1D layer found")


def vanilla_saliency(model, X, cls):
    x = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        score = preds[:, cls]
    grads = tape.gradient(score, x)
    return tf.abs(grads)[..., 0].numpy()


def grad_cam_1d(model, X, cls, out_len):
    conv = last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [conv.output, model.output])
    x = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, cls]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, :], axis=-1)).numpy()
    src = np.linspace(0, 1, cam.shape[1])
    dst = np.linspace(0, 1, out_len)
    return np.array([np.interp(dst, src, c) for c in cam])


def smooth_vec(v, k=9):
    return np.convolve(v, np.ones(k) / k, mode="same")


def norm01(v):
    v = v - v.min()
    return v / v.max() if v.max() > 0 else v


# ---------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAL_DIR.mkdir(parents=True, exist_ok=True)

    grid_1868 = np.load(CACHE / "wn_no_os.npy")
    grid_395 = np.load(CACHE / "wn_with_os.npy")
    grid_600 = np.load(ROOT / "output" / "_cache_processed.npz")["wn"]

    files = sorted(DATA_DIR.glob("*.txt"))
    print(f"{len(files)} spectra in {DATA_DIR}")

    names, raw = [], []
    for p in files:
        wn, ity = parse_xy(p)
        if len(wn) < 10:
            print(f"  SKIP {p.name}: unparseable")
            continue
        names.append(p.stem)
        raw.append((wn, ity))
    n = len(names)

    # Align each spectrum to each grid (raw %T; preprocessing comes per family)
    A1868 = np.array([align_clamp(w, i, grid_1868) for w, i in raw])
    A395 = np.array([align_clamp(w, i, grid_395) for w, i in raw])
    A600 = np.array([align_clamp(w, i, grid_600) for w, i in raw])

    # Per-family preprocessed matrices
    print("Preprocessing (asls baseline runs per spectrum; takes a moment) ...")
    X_snv_1868 = preprocess(A1868, CFG_SNV).astype(np.float32)
    X_snv_395 = preprocess(A395, CFG_SNV).astype(np.float32)
    X_d1_1868 = preprocess(A1868, CFG_D1).astype(np.float32)
    X_leg_1868 = preprocess(A1868, CFG_LEGACY).astype(np.float32)
    X_leg_600 = preprocess(A600, CFG_LEGACY).astype(np.float32)

    # --- Load all model families -------------------------------------------
    print("Loading models ...")
    fam_probs = {}

    legacy_cnn = load_keras_ensemble(sorted((ROOT / "output").glob("fold_*.keras")))
    fam_probs["legacy-cnn"] = cnn_proba(legacy_cnn, X_leg_1868)
    del legacy_cnn

    legacy_rf = load_rf_ensemble(sorted((ROOT / "output").glob("rf_fold_*.pkl")))
    fam_probs["legacy-rf"] = rf_proba(legacy_rf, X_leg_600)
    del legacy_rf

    final_rf = load_rf_ensemble(sorted((EXP_MODELS / "final").glob("rf_fold_*.pkl")))
    fam_probs["final-rf"] = rf_proba(final_rf, X_snv_1868)
    del final_rf

    final_cnn = load_keras_ensemble(sorted((EXP_MODELS / "final").glob("cnn_fold_*.keras")))
    fam_probs["final-cnn"] = cnn_proba(final_cnn, X_snv_1868)

    cnn_d1 = load_keras_ensemble(sorted((EXP_MODELS / "no_os__smooth+d1+snv").glob("*.keras")))
    fam_probs["cnn-d1"] = cnn_proba(cnn_d1, X_d1_1868)
    del cnn_d1

    cnn_os = load_keras_ensemble(sorted((EXP_MODELS / "with_os__norm-snv").glob("*.keras")))
    fam_probs["cnn-os"] = cnn_proba(cnn_os, X_snv_395)
    del cnn_os

    families = ["legacy-cnn", "legacy-rf", "final-rf", "final-cnn", "cnn-d1", "cnn-os"]

    # --- Tables --------------------------------------------------------------
    rows = []
    for i, name in enumerate(names):
        row = {"file": name}
        for fam in families:
            p = fam_probs[fam][i]
            k = int(np.argmax(p))
            row[fam] = SIX[k]
            row[fam + "_conf"] = float(p[k])
        # deployment decision: final-rf + reject rule
        rj = fam_probs["final-rf"][i]
        row["deployment"] = ("UNKNOWN" if rj.max() < REJECT_THRESHOLD
                             else SIX[int(np.argmax(rj))])
        rows.append(row)

    with open(OUT_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# Real-world test — all model families",
          f"\n{n} spectra from `{DATA_DIR}`  |  reject rule: final-rf max prob < "
          f"{REJECT_THRESHOLD} -> UNKNOWN\n",
          "| file | deployment (RF+reject) | " + " | ".join(families) + " |",
          "|---|---|" + "---|" * len(families)]
    for r in rows:
        cells = [f"{r[f]} {r[f + '_conf'] * 100:.0f}%" for f in families]
        md.append(f"| {r['file']} | **{r['deployment']}** | " + " | ".join(cells) + " |")

    # --- Saliency on the deployment CNN ensemble -----------------------------
    print("Saliency maps (final CNN ensemble) ...")
    L = len(grid_1868)
    cams_all = {}
    for i, name in enumerate(names):
        cls = int(np.argmax(fam_probs["final-cnn"][i]))
        conf = float(fam_probs["final-cnn"][i][cls])
        Xi = X_snv_1868[i:i + 1][..., np.newaxis]
        sal = np.mean([vanilla_saliency(m, Xi, cls) for m in final_cnn], axis=0)[0]
        cam = np.mean([grad_cam_1d(m, Xi, cls, L) for m in final_cnn], axis=0)[0]
        sal_p = norm01(smooth_vec(sal, 9))
        cam_p = norm01(cam)
        cams_all[name] = (cam_p, SIX[cls])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(grid_1868, norm01(X_snv_1868[i]), color="black", lw=1.0,
                label="spectrum (snv, norm01)")
        ax.imshow(cam_p[None, :], aspect="auto", cmap="inferno", alpha=0.55,
                  extent=[grid_1868.min(), grid_1868.max(), -0.05, 1.05])
        ax.plot(grid_1868, sal_p, color="cyan", lw=0.8, alpha=0.9,
                label="gradient saliency")
        ax.set_xlim(grid_1868.min(), grid_1868.max())
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("normalized")
        rfp = fam_probs["final-rf"][i]
        dep = ("UNKNOWN" if rfp.max() < REJECT_THRESHOLD
               else SIX[int(np.argmax(rfp))])
        ax.set_title(f"{name}  |  CNN: {SIX[cls]} {conf * 100:.0f}%  |  "
                     f"RF: {SIX[int(np.argmax(rfp))]} {rfp.max() * 100:.0f}%  |  "
                     f"deployment: {dep}")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(SAL_DIR / f"{name}.png", dpi=140)
        plt.close(fig)
        print(f"  {name}: CNN {SIX[cls]} {conf * 100:.0f}% | deployment {dep}")

    # Overview: all Grad-CAMs stacked
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.0 * n), sharex=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        cam_p, cls_name = cams_all[name]
        ax.imshow(cam_p[None, :], aspect="auto", cmap="inferno",
                  extent=[grid_1868.min(), grid_1868.max(), 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(f"{name}\n[{cls_name}]", rotation=0, ha="right",
                      va="center", fontsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    np.atleast_1d(axes)[0].set_title(
        "Grad-CAM importance per sample (final CNN ensemble, predicted class)")
    np.atleast_1d(axes)[0].invert_xaxis()   # sharex: one invert flips all
    fig.tight_layout()
    fig.savefig(OUT_DIR / "saliency_overview.png", dpi=140)
    plt.close(fig)

    md += ["\nSaliency maps: `saliency/<file>.png` (Grad-CAM background + "
           "gradient saliency, final CNN ensemble, predicted class); "
           "`saliency_overview.png` stacks all Grad-CAMs.\n"]
    (OUT_DIR / "predictions.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'predictions.csv'}, predictions.md, "
          f"{n} saliency maps -> {SAL_DIR}")


if __name__ == "__main__":
    main()
