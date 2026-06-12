"""
End-to-end pipeline: extract -> preprocess -> train CNN (5-fold) -> compare.

Stages
------
1. EXTRACT    data/format_data.py aligns FTIR c4 + c8 + OpenSpecy onto a single
              shared wavenumber grid and drops every spectrum that still has a
              NaN after alignment.  Output: X (n_spectra x n_points), integer
              labels, and the wavenumber axis.
2. PREPROCESS preprocess.py applies the (editable) PreprocessConfig below to all
              spectra at once.  Every transform is per-spectrum and leakage-free,
              so it is safe to run before the cross-validation split.
3. TRAIN/TEST models/cnn_model_draft.py provides create_model / train_model /
              test_model.  We shuffle, then run a 5-fold StratifiedKFold; each
              fold trains a fresh CNN and predicts its held-out test split.  The
              out-of-fold predictions are concatenated so every spectrum is
              predicted exactly once by a model that never saw it.
4. COMPARE    compare_predictions() scores true vs predicted labels (accuracy,
              macro-F1, per-class F1, confusion matrix).  It is model-agnostic:
              feed it predictions from the CNN, a random forest, or anything else
              in the same (y_true, y_pred) integer-label format.

Run from the project root:
    py run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn.model_selection import StratifiedKFold

# Make the loader's cm^-1 prints survive a non-UTF8 console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from data.format_data import PlasticIRDataset, POLYMER_CLASSES
from preprocess import preprocess, PreprocessConfig
from models.cnn_model_draft import create_model, train_model, test_model
from models.rf_model import run_rf_cv

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

N_FOLDS = 5
RANDOM_STATE = 0


# ===========================================================================
# EDIT HERE -- preprocessing toggles
# ---------------------------------------------------------------------------
# Flip these to change what runs in stage 2.  Defaults below = raw (control).
# See preprocess.PreprocessConfig for every field and its valid values:
#   smooth      : bool            Savitzky-Golay smoothing
#   baseline    : None|"asls"|"arpls"   baseline correction
#   derivative  : 0|1|2           Savitzky-Golay derivative order (0 = off)
#   normalize   : "none"|"minmax"|"snv"|"l2"
#   region      : None|(lo, hi)   keep only this wavenumber window (cm^-1)
# ===========================================================================
PREPROCESS_CONFIG = PreprocessConfig(
    smooth=True,
    baseline="asls",
    derivative=0,
    normalize="snv",   # original CNN used per-spectrum [0,1] scaling
    region=None,
)
# ===========================================================================


LAB_FOLDER = DATA_DIR / "ftir_real_world_samples"


def extract_data(include_lab=True):
    """Stage 1: load + align all spectra onto the shared wavenumber grid.

    Training sources are FTIR c4 + c8 + OpenSpecy.  The real-world lab spectra
    (LAB_FOLDER/*.txt) are loaded AFTER the grid is decided, so they never
    influence the wavenumber grid and -- crucially -- are returned separately so
    the caller can hold them out of every training fold (TEST-ONLY).

    Returns
    -------
    X     : np.ndarray (n_spectra, n_points)  training absorbance values, no NaNs
    y     : np.ndarray (n_spectra,) int        integer class labels (LABEL_TO_INT)
    wn    : np.ndarray (n_points,)             shared wavenumber axis (cm^-1)
    X_lab : np.ndarray (n_lab, n_points) | None  test-only lab spectra
    y_lab : np.ndarray (n_lab,) int | None        test-only lab labels
    """
    ds = PlasticIRDataset(
        ftir_c4_path=str(DATA_DIR / "FTIR_PLASTIC_c4.csv"),
        ftir_c8_path=str(DATA_DIR / "FTIR_PLASTIC_c8.csv"),
        openspecy_dataset_path=str(DATA_DIR / "openspecy_polymer_dataset.csv"),
        openspecy_metadata_path=str(DATA_DIR / "openspecy_polymer_metadata.csv"),
        openspecy_wavenumbers_path=str(DATA_DIR / "openspecy_wavenumbers.csv"),
    )
    ds.process()
    formatted, wn = ds.get_formatted_data()

    X = np.array([e["intensities"] for e in formatted], dtype=float)
    y = np.array([e["label_int"] for e in formatted], dtype=int)

    # Guard against any label that fell outside the 6 known classes (-1).
    if (y < 0).any():
        bad = sorted({e["label"] for e in formatted if e["label_int"] < 0})
        raise ValueError(f"Unmapped polymer labels in formatted data: {bad}")

    print(f"\nExtracted {X.shape[0]} training spectra x {X.shape[1]} wavenumber points")

    # --- Test-only lab data --------------------------------------------------
    X_lab = y_lab = None
    if include_lab and LAB_FOLDER.exists():
        n_train = len(formatted)                    # snapshot before appending
        ds.add_lab_data(str(LAB_FOLDER))            # aligns lab to the SAME grid
        # Everything appended past n_train is lab; drop any row that aligned to a
        # NaN (range did not cover the grid) or has an unmapped label.
        lab = [e for e in ds.formatted_data[n_train:]
               if e["label_int"] >= 0 and not np.isnan(e["intensities"]).any()]
        if lab:
            X_lab = np.array([e["intensities"] for e in lab], dtype=float)
            y_lab = np.array([e["label_int"] for e in lab], dtype=int)
            print(f"Loaded {X_lab.shape[0]} lab spectra (TEST-ONLY; never trained on)")
        else:
            print("No usable lab spectra after alignment -- skipping lab test.")
    elif include_lab:
        print(f"Lab folder not found: {LAB_FOLDER} -- skipping lab test.")

    return X, y, np.asarray(wn, dtype=float), X_lab, y_lab


def preprocess_data(X, wn, config=PREPROCESS_CONFIG):
    """Stage 2: run all spectra through the configured preprocessing pipeline.

    Returns the processed matrix Xp and the (possibly region-trimmed) wavenumber
    axis.  Preprocessing here is per-spectrum / leakage-free, so doing it before
    the CV split introduces no train/test leakage.
    """
    print("\nPreprocessing with config:")
    for field, value in vars(config).items():
        print(f"  {field:<22} = {value}")

    Xp, wn_out = preprocess(X, config, wavenumbers=wn)
    print(f"Preprocessed shape: {Xp.shape}")
    return Xp, wn_out


def run_cnn_cv(X, y, X_lab=None, n_folds=N_FOLDS, seed=RANDOM_STATE, save_models=True):
    """Stage 3 (CNN): shuffle, 5-fold StratifiedKFold, train+test a CNN per fold.

    Each training spectrum is predicted exactly once, by the fold model that held
    it out.  The internal 70/30 train/validation split (for early stopping) lives
    inside train_model and is unchanged.

    Lab data (X_lab) is used for TESTING ONLY: it never enters a training fold.
    Each of the 5 fold models predicts the full lab set, and their softmax
    probabilities are averaged into a single 5-model ensemble prediction.

    Returns a dict:
        y_true, y_pred, y_proba : concatenated out-of-fold arrays (training data)
        fold_acc                : list[float] per-fold accuracy
        lab_pred, lab_proba     : ensemble lab prediction / probabilities (or None)
    """
    # Conv1D expects (samples, timesteps, channels) -- add the channel axis.
    X = X[..., np.newaxis]
    if X_lab is not None:
        X_lab = X_lab[..., np.newaxis]

    # Shuffle once up front; the SAME permutation is applied to X and y so each
    # spectrum keeps its label.  StratifiedKFold also shuffles internally, but an
    # explicit shuffle keeps behaviour obvious and order-independent.
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    if save_models:
        OUTPUT_DIR.mkdir(exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    y_true_all, y_pred_all, y_proba_all, fold_acc = [], [], [], []
    lab_proba_sum = None
    n_classes = len(POLYMER_CLASSES)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- CNN fold {fold}/{n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = create_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            n_classes=n_classes,
            seed=seed)

        train_model(model, X_train, y_train, seed=seed)
        y_pred, y_proba = test_model(model, X_test)

        acc = skm.accuracy_score(y_test, y_pred)
        fold_acc.append(acc)
        print(f"CNN fold {fold} accuracy: {acc:.4f}")

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)

        if X_lab is not None:
            _, lab_proba = test_model(model, X_lab)
            lab_proba_sum = lab_proba if lab_proba_sum is None else lab_proba_sum + lab_proba

        if save_models:
            path = OUTPUT_DIR / f"cnn_fold_{fold}.keras"
            model.save(path)
            print(f"Saved model to {path}")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)

    print(f"\nCNN per-fold accuracy: "
          + ", ".join(f"{a:.4f}" for a in fold_acc))
    print(f"CNN mean accuracy: {np.mean(fold_acc):.4f} "
          f"(+/- {np.std(fold_acc):.4f})")

    lab_pred = lab_proba = None
    if lab_proba_sum is not None:
        lab_proba = lab_proba_sum / n_folds
        lab_pred = np.argmax(lab_proba, axis=1)

    return {"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
            "fold_acc": fold_acc, "lab_pred": lab_pred, "lab_proba": lab_proba}


# ===========================================================================
# Stage 4 -- general comparison (model-agnostic)
# ===========================================================================

def compare_predictions(
    y_true,
    y_pred,
    class_names=POLYMER_CLASSES,
    model_name="model",
    save_confusion=True,
    output_dir=OUTPUT_DIR,
):
    """Score predicted vs true labels for ANY classifier.

    Works with integer labels (0..n_classes-1) as produced by the CNN's
    test_model() and by the random-forest model.  Future models only need to
    emit predictions in this same format to be comparable.

    Parameters
    ----------
    y_true, y_pred : array-like of int
        True and predicted class indices.
    class_names : list[str]
        Names indexed by class id, for display / the confusion matrix.
    model_name : str
        Used in printout, plot title, and the saved PNG filename.
    save_confusion : bool
        If True, write a confusion-matrix PNG to output_dir.
    output_dir : Path

    Returns
    -------
    dict with keys:
        accuracy        float
        macro_f1        float
        per_class_f1    dict[str, float]
        confusion       np.ndarray (n_classes x n_classes), rows=true
        class_names     list[str]
        model_name      str
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    accuracy = skm.accuracy_score(y_true, y_pred)
    macro_f1 = skm.f1_score(y_true, y_pred, labels=labels,
                            average="macro", zero_division=0)
    # average=None returns one F1 per class as an ndarray.  Wrap in atleast_1d so
    # static type checkers see an iterable ndarray (not the float|ndarray union
    # the stubs declare) -- runtime behaviour is unchanged.
    per_class = np.atleast_1d(skm.f1_score(y_true, y_pred, labels=labels,
                                           average=None, zero_division=0))
    per_class_f1 = dict(zip(class_names, (float(v) for v in per_class)))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n=== {model_name} ===")
    print(f"  accuracy : {accuracy:.4f}")
    print(f"  macro-F1 : {macro_f1:.4f}")
    print(f"  per-class F1: "
          + ", ".join(f"{k}={v:.3f}" for k, v in per_class_f1.items()))

    if save_confusion:
        output_dir.mkdir(exist_ok=True)
        path = output_dir / f"confusion_{model_name}.png"
        _plot_confusion(cm, class_names,
                        f"{model_name}  acc={accuracy:.3f}  macroF1={macro_f1:.3f}",
                        path)
        print(f"  saved confusion matrix to {path}")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "confusion": cm,
        "class_names": list(class_names),
        "model_name": model_name,
    }


def _plot_confusion(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

def _summary_row(name, metrics):
    return f"  {name:<5} acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}"


def main():
    # --- Stage 1: extract (training sources + test-only lab data) -----------
    X, y, wn, X_lab, y_lab = extract_data(include_lab=True)

    # --- Stage 2: preprocess ------------------------------------------------
    # Lab data is preprocessed with the SAME config and the SAME (full) wn axis,
    # so it ends up on identical columns to the training data.
    Xp, _ = preprocess_data(X, wn, PREPROCESS_CONFIG)
    Xp_lab = None
    if X_lab is not None:
        Xp_lab = preprocess(X_lab, PREPROCESS_CONFIG, wavenumbers=wn)[0]
        print(f"Lab (test-only) preprocessed shape: {Xp_lab.shape}")

    results = {}

    # --- Stage 3 + 4: CNN ---------------------------------------------------
    cnn = run_cnn_cv(Xp, y, X_lab=Xp_lab)
    results["cnn_cv"] = compare_predictions(cnn["y_true"], cnn["y_pred"],
                                            model_name="cnn_cv")
    if Xp_lab is not None:
        results["cnn_lab"] = compare_predictions(y_lab, cnn["lab_pred"],
                                                 model_name="cnn_lab")

    # --- Stage 3 + 4: Random Forest -----------------------------------------
    rf = run_rf_cv(Xp, y, X_lab=Xp_lab, output_dir=OUTPUT_DIR)
    results["rf_cv"] = compare_predictions(rf["y_true"], rf["y_pred"],
                                           model_name="rf_cv")
    if Xp_lab is not None:
        results["rf_lab"] = compare_predictions(y_lab, rf["lab_pred"],
                                                model_name="rf_lab")

    # --- Head-to-head summary ----------------------------------------------
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD SUMMARY")
    print("=" * 60)
    print("Cross-validation (held-out folds of the training data):")
    print(_summary_row("CNN", results["cnn_cv"]))
    print(_summary_row("RF", results["rf_cv"]))
    if "cnn_lab" in results:
        print("Real-world lab data (test-only generalization, 5-model ensemble):")
        print(_summary_row("CNN", results["cnn_lab"]))
        print(_summary_row("RF", results["rf_lab"]))


if __name__ == "__main__":
    main()
