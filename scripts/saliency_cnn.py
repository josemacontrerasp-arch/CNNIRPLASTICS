"""
Saliency / explainability analysis for EVERY trained 1-D CNN family.

For each CNN model family in the repo we produce heatmaps showing which
wavenumber regions drive the decision, using two complementary methods
ensembled over that family's folds:

  1. Vanilla gradient saliency  -- |d p(class) / d input| per wavenumber.
  2. Grad-CAM (1-D)             -- gradients of the class score w.r.t. the last
     Conv1D feature maps, GAP-pooled into channel weights, ReLU-ed weighted sum,
     upsampled back to the spectrum length.

Each family is fed the EXACT preprocessed training matrix it was trained on
(the cached experiments_output/cache/Xp_*.npy), so the maps reflect what the
deployed model actually sees -- not a re-derived approximation.

Families covered (CNN only; gradient methods need a differentiable model):
  final-cnn   experiments_output/models/final            no-OS, SNV,            1868 pts
  cnn-d1      .../no_os__smooth+d1+snv                    no-OS, smooth+d1+SNV,  1868 pts
  cnn-os      .../with_os__norm-snv                       with-OS, SNV,          395 pts (804-3168)
  legacy-cnn  output/fold_*.keras                         min-max (its original pipeline), 1868 pts
  (no_os__norm-snv is config-identical to final-cnn, so it is not duplicated.)

For each family x class we average maps over up-to-N correctly-classified
training spectra, overlay them on the mean spectrum, mark the known diagnostic
IR bands, and run two reproducible artifact checks (CO2 band, high-wavenumber
reliance).

Outputs -> results/saliency/<family>/
  saliency_<CLASS>.png   per-class overlay
  saliency_overview.png  six classes stacked (Grad-CAM)
  saliency_analysis.md   per-family written analysis incl. band overlap
plus results/saliency/INDEX.md linking all families.

No training happens here -- inference + gradients on saved models only.
"""
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
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "experiments_output" / "cache"
EXP = ROOT / "experiments_output" / "models"
OUT_ROOT = ROOT / "results" / "saliency"
SIX = ["HDPE", "LDPE", "PP", "PS", "PVC", "PET"]      # model softmax order
LABEL_TO_INT = {c: i for i, c in enumerate(SIX)}
N_PER_CLASS = 25

DIAGNOSTIC_BANDS = {
    "HDPE": [2915, 2848, 1471, 730, 719],
    "LDPE": [2915, 2848, 1465, 1377, 730, 719],
    "PP":   [2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840],
    "PS":   [3026, 2920, 1601, 1492, 1452, 1027, 753, 696],
    "PVC":  [2912, 1427, 1331, 1254, 960, 690, 615],
    "PET":  [1715, 1409, 1241, 1094, 1017, 871, 722],
}


def minmax(X):
    lo = X.min(axis=1, keepdims=True)
    rng = X.max(axis=1, keepdims=True) - lo
    rng[rng == 0] = 1.0
    return (X - lo) / rng


# ---- family definitions: how to get (X, labels, grid, models) for each -----
def family_specs():
    specs = []

    # final-cnn: deployment CNN, no-OS, SNV, 1868 pts
    specs.append(dict(
        name="final-cnn", slug="final-cnn",
        desc="deployment CNN -- no-OpenSpecy, SNV, 1868 pts",
        X=lambda: np.load(CACHE / "Xp_no_os_norm-snv.npy"),
        labels=lambda: np.load(CACHE / "labels_no_os.npy", allow_pickle=True),
        grid=lambda: np.load(CACHE / "wn_no_os.npy"),
        models=sorted((EXP / "final").glob("cnn_fold_*.keras")),
    ))
    # cnn-d1: no-OS, smooth + 1st derivative + SNV, 1868 pts
    specs.append(dict(
        name="cnn-d1", slug="cnn-d1",
        desc="no-OpenSpecy, smooth + 1st-derivative + SNV, 1868 pts",
        X=lambda: np.load(CACHE / "Xp_no_os_smooth+d1+snv.npy"),
        labels=lambda: np.load(CACHE / "labels_no_os.npy", allow_pickle=True),
        grid=lambda: np.load(CACHE / "wn_no_os.npy"),
        models=sorted((EXP / "no_os__smooth+d1+snv").glob("*.keras")),
    ))
    # cnn-os: with-OS, SNV, 395 pts (804-3168)
    specs.append(dict(
        name="cnn-os", slug="cnn-os",
        desc="with-OpenSpecy, SNV, 395 pts (804-3168 cm^-1)",
        X=lambda: np.load(CACHE / "Xp_with_os_norm-snv.npy"),
        labels=lambda: np.load(CACHE / "labels_with_os.npy", allow_pickle=True),
        grid=lambda: np.load(CACHE / "wn_with_os.npy"),
        models=sorted((EXP / "with_os__norm-snv").glob("*.keras")),
    ))
    # legacy-cnn: output/fold_*.keras, min-max on the raw aligned 1868 grid
    specs.append(dict(
        name="legacy-cnn", slug="legacy-cnn",
        desc="legacy output/fold_*.keras -- per-spectrum min-max, 1868 pts "
             "(original pre-OpenSpecy pipeline)",
        X=lambda: minmax(np.load(CACHE / "X_no_os.npy")),
        labels=lambda: np.load(CACHE / "labels_no_os.npy", allow_pickle=True),
        grid=lambda: np.load(CACHE / "wn_no_os.npy"),
        models=sorted((ROOT / "output").glob("fold_*.keras")),
    ))
    return specs


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


def smooth(v, k=9):
    return np.convolve(v, np.ones(k) / k, mode="same") if k > 1 else v


def norm01(v):
    v = v - v.min()
    return v / v.max() if v.max() > 0 else v


def top_band_centers(grid, profile, n_peaks=8, min_sep_cm=40):
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
    peaks = top_band_centers(grid, cam_profile)
    # only bands within the family's grid range are reachable
    lo, hi = grid.min(), grid.max()
    bands = [b for b in DIAGNOSTIC_BANDS[label] if lo <= b <= hi]
    hits = [b for b in bands if any(abs(b - p) <= tol for p in peaks)]
    return peaks, bands, hits


# ---------------------------------------------------------------- plotting
def plot_class(grid, mean_spec, sal, cam, label, bands, hits, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, mean_spec, color="black", lw=1.0, label="mean spectrum (norm)")
    ax.imshow(cam[None, :], aspect="auto", cmap="inferno", alpha=0.55,
              extent=[grid.min(), grid.max(), -0.05, 1.05])
    ax.plot(grid, sal, color="cyan", lw=0.8, alpha=0.9, label="gradient saliency")
    for b in DIAGNOSTIC_BANDS[label]:
        if not (grid.min() <= b <= grid.max()):
            continue
        c = "lime" if b in hits else "red"
        ax.axvline(b, color=c, ls=":", lw=1.0, alpha=0.8)
    ax.set_xlim(grid.min(), grid.max())
    ax.invert_xaxis()
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("normalized")
    ax.set_title(f"{label}: Grad-CAM (bg) + gradient saliency | "
                 f"in-range bands hit {len(hits)}/{len(bands)} "
                 f"(green=hit, red=miss)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"saliency_{label}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_overview(grid, cams, out_dir, fam_name):
    fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
    for ax, label in zip(axes, SIX):
        cam = cams[label]
        ax.imshow(cam[None, :], aspect="auto", cmap="inferno",
                  extent=[grid.min(), grid.max(), 0, 1])
        for b in DIAGNOSTIC_BANDS[label]:
            if grid.min() <= b <= grid.max():
                ax.axvline(b, color="cyan", ls=":", lw=0.7, alpha=0.7)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[0].set_title(f"{fam_name}: per-class Grad-CAM (dotted = diagnostic bands)")
    for ax in axes:
        ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "saliency_overview.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- per family
def run_family(spec):
    out_dir = OUT_ROOT / spec["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not spec["models"]:
        print(f"  [skip] no models for {spec['name']}")
        return None
    X = np.asarray(spec["X"](), dtype=np.float32)
    labels = np.asarray(spec["labels"]())
    grid = np.asarray(spec["grid"](), dtype=float)

    # Reconcile labels to X: the with-OS preprocessed matrix is the six-class
    # subset (bioplastics dropped) while labels_with_os covers all classes.
    # Masking to the six classes restores the alignment (same original order).
    if len(labels) != len(X):
        six_mask = np.isin(labels, SIX)
        if six_mask.sum() == len(X):
            labels = labels[six_mask]
        else:
            raise ValueError(f"{spec['name']}: cannot align labels "
                             f"({len(labels)}, six={six_mask.sum()}) to X ({len(X)})")
    L = len(grid)
    models = [tf.keras.models.load_model(p) for p in spec["models"]]
    print(f"\n=== {spec['name']} === ({spec['desc']})")
    print(f"  X {X.shape}, grid {L} pts {grid.min():.0f}-{grid.max():.0f}, "
          f"{len(models)} folds")

    md = [f"# Saliency -- {spec['name']}\n", f"_{spec['desc']}_\n",
          f"Ensemble of {len(models)} folds; maps averaged over up to "
          f"{N_PER_CLASS} correctly-classified training spectra per class.\n",
          "## Per-class diagnostic-band overlap\n",
          "| Class | Grad-CAM peaks (cm⁻¹) | In-range bands | Hit |",
          "|---|---|---|---:|"]
    cams = {}
    for label in SIX:
        cls = LABEL_TO_INT[label]
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            cams[label] = np.zeros(L)
            md.append(f"| {label} | (no samples) | - | - |")
            continue
        Xc0 = X[idx]
        probs = np.mean([m.predict(Xc0[..., None], verbose=0) for m in models], axis=0)
        correct = idx[np.argmax(probs, axis=1) == cls][:N_PER_CLASS]
        if len(correct) == 0:
            correct = idx[:N_PER_CLASS]
        Xc = X[correct][..., None].astype(np.float32)
        mean_spec = norm01(X[correct].mean(axis=0))

        sal = np.mean([vanilla_saliency(m, Xc, cls) for m in models], axis=0)
        cam = np.mean([grad_cam_1d(m, Xc, cls, L) for m in models], axis=0)
        sal_p = norm01(smooth(sal.mean(axis=0), 9))
        cam_p = norm01(cam.mean(axis=0))
        cams[label] = cam_p

        peaks, bands, hits = overlap_report(grid, cam_p, label)
        plot_class(grid, mean_spec, sal_p, cam_p, label, bands, hits, out_dir)
        md.append(f"| {label} | {', '.join(f'{p:.0f}' for p in peaks)} | "
                  f"{', '.join(str(b) for b in bands)} | {len(hits)}/{len(bands)} |")
        print(f"  {label}: {len(correct)} samples, band hits {len(hits)}/{len(bands)}")

    plot_overview(grid, cams, out_dir, spec["name"])

    # artifact checks (only if grid covers the regions)
    extras = []
    if grid.min() <= 2349 <= grid.max():
        ci = int(np.argmin(np.abs(grid - 2349)))
        win = slice(max(0, ci - 15), ci + 15)
        co2 = [l for l in SIX if cams[l][win].max() >= 0.5]
        extras.append(f"* **CO₂ band (~2349 cm⁻¹)** strongly weighted by: "
                      f"{', '.join(co2) if co2 else 'none'}.")
    hi = grid >= 2800
    if hi.any():
        hf = {l: float(cams[l][hi].sum() / cams[l].sum()) for l in SIX}
        extras.append("* **High-wavenumber reliance (≥2800 cm⁻¹):** "
                      + ", ".join(f"{l} {f*100:.0f}%" for l, f in hf.items()) + ".")
    if extras:
        md += ["\n## Reproducible artifact checks\n", *extras]

    (out_dir / "saliency_analysis.md").write_text("\n".join(md), encoding="utf-8")
    for m in models:
        del m
    return {"name": spec["name"], "slug": spec["slug"], "desc": spec["desc"]}


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = []
    for spec in family_specs():
        r = run_family(spec)
        if r:
            done.append(r)

    idx = ["# CNN Saliency Analysis -- all families\n",
           "Gradient saliency + 1-D Grad-CAM per CNN family, each using its own "
           "training preprocessing and wavenumber grid.\n",
           "| Family | Configuration | Folder |", "|---|---|---|"]
    for r in done:
        idx.append(f"| {r['name']} | {r['desc']} | "
                   f"[`{r['slug']}/`]({r['slug']}/saliency_analysis.md) |")
    idx += ["\nEach folder has `saliency_<CLASS>.png` (overlay), "
            "`saliency_overview.png` (six classes stacked), and "
            "`saliency_analysis.md` (band-overlap + artifact checks)."]
    (OUT_ROOT / "INDEX.md").write_text("\n".join(idx), encoding="utf-8")
    print(f"\nWrote {OUT_ROOT/'INDEX.md'} covering {len(done)} families")


if __name__ == "__main__":
    main()
