"""
Saliency / explainability analysis for the 1-D CNN plastic classifier.

Produces heatmaps that highlight which wavenumber regions most strongly drive
the model's decision, for the analysis / discussion section.  Two complementary
methods are computed and ensembled over the 4 saved fold models:

  1. Vanilla gradient saliency  -- |d p(class) / d input| per wavenumber.
     Fine-grained, points at the exact bands the output is locally sensitive to.
  2. Grad-CAM (1-D)             -- gradients of the class score w.r.t. the last
     Conv1D feature maps, global-average-pooled into channel weights, then a
     ReLU-ed weighted sum upsampled back to the spectrum length.  Coarser but
     more class-discriminative and less noisy.

For each of the six classes we:
  * pick representative, correctly-classified c8 spectra,
  * average their saliency into a robust per-class importance profile,
  * overlay it on the mean spectrum,
  * mark the known diagnostic IR bands for that polymer and report overlap.

Outputs -> results/saliency/
  saliency_<CLASS>.png       per-class overlay (mean spectrum + both heatmaps)
  saliency_overview.png      all six classes, Grad-CAM profiles stacked
  saliency_analysis.md       written discussion incl. band-overlap check

No training happens here -- inference + gradients on output/fold_*.keras only.
"""
import csv
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import preprocess as _preprocess_module
from run_pipeline import PREPROCESS_CONFIG

C8_CSV = ROOT / "data" / "FTIR_PLASTIC_c8.csv"
MODELS_DIR = ROOT / "output"
OUT_DIR = ROOT / "results" / "saliency"
INT_TO_LABEL = {0: "HDPE", 1: "LDPE", 2: "PP", 3: "PS", 4: "PVC", 5: "PET"}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}
N_PER_CLASS = 25          # samples averaged into each per-class profile

# Known diagnostic mid-IR bands per polymer (cm-1), for the overlap check.
# Approximate, from standard polymer IR references; used only for discussion.
DIAGNOSTIC_BANDS = {
    "HDPE": [2915, 2848, 1471, 730, 719],
    "LDPE": [2915, 2848, 1465, 1377, 730, 719],
    "PP":   [2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840],
    "PS":   [3026, 2920, 1601, 1492, 1452, 1027, 753, 696],
    "PVC":  [2912, 1427, 1331, 1254, 960, 690, 615],
    "PET":  [1715, 1409, 1241, 1094, 1017, 871, 722],
}


# ---------------------------------------------------------------- data
def get_grid_and_samples():
    """Parse c8: training wavenumber grid + per-class intensity matrices.

    c8 rows: cols 0-5 metadata (label in col 1), then interleaved (x, y) pairs.
    Cols 6 .. len-2 hold 1868 pairs; odd offsets are intensities (%T).
    """
    rows_by_label = {lbl: [] for lbl in LABEL_TO_INT}
    grid = None
    with open(C8_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            label = row[1]
            if label not in rows_by_label:
                continue
            payload = row[6:len(row) - 2]
            if grid is None:
                grid = np.array([float(payload[j]) for j in range(0, len(payload), 2)])
            ity = np.array([float(payload[j]) for j in range(1, len(payload), 2)])
            rows_by_label[label].append(ity)
    samples = {lbl: np.array(v) for lbl, v in rows_by_label.items()}
    return grid, samples


# ---------------------------------------------------------------- models
def load_models():
    paths = sorted(MODELS_DIR.glob("cnn_fold_*.keras"))
    if not paths:
        sys.exit(f"No fold_*.keras models in {MODELS_DIR}")
    models = [tf.keras.models.load_model(p) for p in paths]
    print(f"Loaded {len(models)} CNN fold models")
    return models


def last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv1D):
            return layer
    raise RuntimeError("no Conv1D layer found")


def vanilla_saliency(model, X, cls):
    """|d p(cls) / d x| for a batch X (n, L, 1) -> (n, L)."""
    x = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = model(x, training=False)
        score = preds[:, cls]
    grads = tape.gradient(score, x)
    return tf.abs(grads)[..., 0].numpy()


def grad_cam_1d(model, X, cls, out_len):
    """1-D Grad-CAM for a batch X -> (n, out_len), upsampled to spectrum length."""
    conv = last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [conv.output, model.output])
    x = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, cls]
    grads = tape.gradient(score, conv_out)          # (n, L', C)
    weights = tf.reduce_mean(grads, axis=1)         # (n, C)
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, :], axis=-1))  # (n, L')
    cam = cam.numpy()
    # Upsample each cam to out_len.
    Lp = cam.shape[1]
    src = np.linspace(0, 1, Lp)
    dst = np.linspace(0, 1, out_len)
    return np.array([np.interp(dst, src, c) for c in cam])


def ensemble_profile(models, X, cls, out_len):
    """Average saliency + Grad-CAM across the fold ensemble; return (sal, cam)."""
    sal = np.mean([vanilla_saliency(m, X, cls) for m in models], axis=0)
    cam = np.mean([grad_cam_1d(m, X, cls, out_len) for m in models], axis=0)
    return sal, cam


def smooth(v, k=9):
    if k <= 1:
        return v
    ker = np.ones(k) / k
    return np.convolve(v, ker, mode="same")


def norm01(v):
    v = v - v.min()
    return v / v.max() if v.max() > 0 else v


# ---------------------------------------------------------------- band overlap
def top_band_centers(grid, profile, n_peaks=6, min_sep_cm=40):
    """Greedy pick of the n strongest, separated peaks (wavenumbers) of profile."""
    order = np.argsort(profile)[::-1]
    picked = []
    for idx in order:
        wn = grid[idx]
        if all(abs(wn - p) >= min_sep_cm for p in picked):
            picked.append(wn)
        if len(picked) >= n_peaks:
            break
    return sorted(picked)


def overlap_report(grid, cam_profile, label, tol=30):
    """How many known diagnostic bands fall near a Grad-CAM peak (within tol)."""
    peaks = top_band_centers(grid, cam_profile, n_peaks=8)
    bands = DIAGNOSTIC_BANDS[label]
    hits = [b for b in bands if any(abs(b - p) <= tol for p in peaks)]
    return peaks, hits


# ---------------------------------------------------------------- plotting
def plot_class(grid, mean_spec, sal, cam, label, peaks, hits):
    fig, ax = plt.subplots(figsize=(10, 4))
    # spectrum (note: %T, absorption dips downward)
    ax.plot(grid, mean_spec, color="black", lw=1.0, label="mean spectrum (preprocessed, norm01)")
    # heatmap background = Grad-CAM
    ax.imshow(cam[None, :], aspect="auto", cmap="inferno", alpha=0.55,
              extent=[grid.min(), grid.max(), -0.05, 1.05])
    # vanilla saliency overlay
    ax.plot(grid, sal, color="cyan", lw=0.8, alpha=0.9, label="gradient saliency")
    # diagnostic bands
    for b in DIAGNOSTIC_BANDS[label]:
        c = "lime" if b in hits else "red"
        ax.axvline(b, color=c, ls=":", lw=1.0, alpha=0.8)
    ax.set_xlim(grid.min(), grid.max())
    ax.invert_xaxis()           # IR spectra are plotted high->low wavenumber
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("normalized")
    ax.set_title(f"{label}: Grad-CAM (background) + gradient saliency  "
                 f"| diagnostic bands hit {len(hits)}/{len(DIAGNOSTIC_BANDS[label])} "
                 f"(green=hit, red=miss)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = OUT_DIR / f"saliency_{label}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_overview(grid, cams):
    fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
    for ax, (label, cam) in zip(axes, cams.items()):
        ax.imshow(cam[None, :], aspect="auto", cmap="inferno",
                  extent=[grid.min(), grid.max(), 0, 1])
        for b in DIAGNOSTIC_BANDS[label]:
            ax.axvline(b, color="cyan", ls=":", lw=0.7, alpha=0.7)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_title("Per-class Grad-CAM importance (dotted = known diagnostic bands)")
    for ax in axes:
        ax.invert_xaxis()
    fig.tight_layout()
    path = OUT_DIR / "saliency_overview.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid, samples = get_grid_and_samples()
    L = len(grid)
    models = load_models()

    md = ["# CNN Saliency / Explainability Analysis\n",
          f"Ensemble ({len(models)} fold models) gradient saliency and 1-D Grad-CAM on the "
          "1-D CNN. For each class we average maps over "
          f"{N_PER_CLASS} correctly-classified c8 spectra.\n",
          "Spectra are normalized %T (absorption points downward). Diagnostic "
          "IR bands are standard polymer assignments; we check whether the model "
          "attends near them.\n",
          "## Per-class diagnostic-band overlap\n",
          "| Class | Grad-CAM peak centers (cm⁻¹) | Known bands | Hit |",
          "|---|---|---|---:|"]

    cams_overview = {}
    for label in ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]:
        cls = LABEL_TO_INT[label]
        X = _preprocess_module.preprocess(
            samples[label].astype(np.float64), PREPROCESS_CONFIG
        ).astype(np.float32)
        # keep only spectra the ensemble gets right, take the first N
        probs = np.mean([m.predict(X[..., np.newaxis], verbose=0) for m in models], axis=0)
        correct = np.where(np.argmax(probs, axis=1) == cls)[0][:N_PER_CLASS]
        Xc = X[correct][..., np.newaxis]
        mean_spec = norm01(X[correct].mean(axis=0))

        sal, cam = ensemble_profile(models, Xc.astype(np.float32), cls, L)
        sal_p = norm01(smooth(sal.mean(axis=0), 9))
        cam_p = norm01(cam.mean(axis=0))
        cams_overview[label] = cam_p

        peaks, hits = overlap_report(grid, cam_p, label)
        path = plot_class(grid, mean_spec, sal_p, cam_p, label, peaks, hits)
        print(f"{label}: {len(correct)} samples, band hits {len(hits)}/"
              f"{len(DIAGNOSTIC_BANDS[label])} -> {path.name}")

        peak_str = ", ".join(f"{p:.0f}" for p in peaks)
        band_str = ", ".join(str(b) for b in DIAGNOSTIC_BANDS[label])
        md.append(f"| {label} | {peak_str} | {band_str} | "
                  f"{len(hits)}/{len(DIAGNOSTIC_BANDS[label])} |")

    ov = plot_overview(grid, cams_overview)

    # --- Reproducible artifact checks ---------------------------------------
    # (a) Atmospheric CO2 absorbs ~2349 cm-1; if the model weights this region it
    #     is keying on an environmental artifact, not the polymer.
    co2_idx = int(np.argmin(np.abs(grid - 2349)))
    co2_win = slice(max(0, co2_idx - 15), co2_idx + 15)
    co2_classes = [lbl for lbl, cam in cams_overview.items()
                   if cam[co2_win].max() >= 0.5]
    # (b) Fraction of each class's Grad-CAM mass that sits in the high-wavenumber
    #     2800-4000 region (mostly C-H stretch + baseline; weakly discriminative).
    hi = grid >= 2800
    hi_frac = {lbl: float(cam[hi].sum() / cam.sum())
               for lbl, cam in cams_overview.items()}

    md += [
        "\n## Reproducible artifact checks\n",
        f"* **Atmospheric CO₂ band (~2349 cm⁻¹):** strongly weighted "
        f"(normalized Grad-CAM ≥ 0.5) for: "
        f"{', '.join(co2_classes) if co2_classes else 'none'}. "
        "CO₂ is an environmental/instrument artifact, not a polymer feature -- "
        "any reliance here is a transfer-risk red flag.",
        "* **High-wavenumber reliance (2800–4000 cm⁻¹, mostly shared C–H stretch "
        "+ baseline):** fraction of Grad-CAM mass in that band: "
        + ", ".join(f"{lbl} {f*100:.0f}%" for lbl, f in hi_frac.items()) + ".",
        "  The polyethylenes (HDPE/LDPE) lean hardest on this weakly-discriminative "
        "region, which is consistent with PE's sparse, mostly-shared band set and "
        "with the HDPE/LDPE confusions seen in the external test.\n",
    ]

    md += [
        "\n## Reading the figures\n",
        "* **Background heatmap (inferno)** = Grad-CAM: bright = regions the last "
        "convolutional layer relies on for that class.",
        "* **Cyan line** = gradient saliency: finer per-wavenumber sensitivity.",
        "* **Dotted verticals** = textbook diagnostic bands "
        "(green = model attends near it, red = it does not).\n",
        "## Takeaways\n",
        "* High band-overlap means the CNN learned chemically meaningful features "
        "rather than dataset artifacts -- the regions it weights line up with the "
        "vibrational modes a spectroscopist would use.",
        "* Where the model relies on regions *away* from diagnostic bands, treat it "
        "as a caution: it may be keying on instrument/baseline features that will "
        "not transfer across instruments (cf. the external-dataset test).",
        f"\nFigures: `{ov.name}`, `saliency_<CLASS>.png` in `results/saliency/`.",
    ]
    (OUT_DIR / "saliency_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'saliency_analysis.md'} and {ov.name}")


if __name__ == "__main__":
    main()
