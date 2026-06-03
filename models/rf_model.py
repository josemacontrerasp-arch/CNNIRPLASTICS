import sys
import pickle
from pathlib import Path
import numpy as np
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Ensure we can import from the project root (for the standalone main() below).
sys.path.append(str(Path(__file__).parent.parent))


def run_rf_cv(X, y, X_lab=None, n_folds=5, seed=0, n_estimators=100,
              n_classes=6, save_models=True, save_importances=True,
              output_dir=None):
    """Random Forest 5-fold StratifiedKFold on ALREADY-PREPROCESSED data.

    This function does NO extraction and NO normalization.  It expects the exact
    same preprocessed feature matrix the CNN is trained on -- run_pipeline.py is
    the single preprocessing authority (extract_data -> preprocess_data) and
    feeds the result here, so both models see identical inputs.

    Parameters
    ----------
    X : np.ndarray (n_spectra, n_points)
        Preprocessed, flat spectra (no channel axis).
    y : np.ndarray (n_spectra,) int
        Integer class labels.
    X_lab : np.ndarray (n_lab, n_points) | None
        Preprocessed test-only lab spectra.  Never used for training; each fold
        model predicts it and the per-fold predict_proba is averaged into a
        single 5-model ensemble prediction.
    n_folds, seed, n_estimators, n_classes : int
        Standard CV / RF hyperparameters.  n_classes sizes the probability
        columns so they align with the CNN's softmax columns.
    save_models, save_importances : bool
        Persist per-fold .pkl models / mean feature importances + summary
        (the latter keeps models/rf_analysis.py working).
    output_dir : str | Path | None
        Where to save artifacts (defaults to ./output).

    Returns
    -------
    dict with the same shape as the CNN's run_cnn_cv:
        y_true, y_pred, y_proba : concatenated out-of-fold arrays
        fold_acc                : list[float] per-fold accuracy
        lab_pred, lab_proba     : ensemble lab prediction / probabilities (or None)
    """
    if output_dir is None:
        output_dir = Path("output")
    output_dir = Path(output_dir)
    if save_models or save_importances:
        output_dir.mkdir(exist_ok=True)

    # Shuffle once; the SAME permutation goes to X and y (matches the CNN path).
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    y_true_all, y_pred_all, y_proba_all, fold_acc, importances = [], [], [], [], []
    lab_proba_sum = None

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- RF fold {fold}/{n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # n_estimators=100 is standard and fast; n_jobs=-1 uses all CPU cores.
        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        # Map predict_proba (ordered by rf.classes_) into the full 0..n_classes-1
        # column space so probabilities line up with the CNN's softmax columns.
        proba = np.zeros((len(X_test), n_classes))
        proba[:, rf.classes_] = rf.predict_proba(X_test)

        acc = skm.accuracy_score(y_test, y_pred)
        fold_acc.append(acc)
        print(f"RF fold {fold} accuracy: {acc:.4f}")

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_proba_all.append(proba)
        importances.append(rf.feature_importances_)

        if X_lab is not None:
            lab_p = np.zeros((len(X_lab), n_classes))
            lab_p[:, rf.classes_] = rf.predict_proba(X_lab)
            lab_proba_sum = lab_p if lab_proba_sum is None else lab_proba_sum + lab_p

        if save_models:
            model_path = output_dir / f"rf_fold_{fold}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(rf, f)
            print(f"Saved model to {model_path}")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_proba = np.concatenate(y_proba_all)

    mean_acc, std_acc = float(np.mean(fold_acc)), float(np.std(fold_acc))
    print(f"\nRF per-fold accuracy: " + ", ".join(f"{a:.4f}" for a in fold_acc))
    print(f"RF mean accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

    if save_importances:
        mean_importances = np.mean(importances, axis=0)
        np.save(output_dir / "rf_feature_importances.npy", mean_importances)
        with open(output_dir / "rf_summary.txt", "w", encoding="utf-8") as f:
            f.write("Random Forest 5-Fold Stratified Cross-Validation Summary\n")
            f.write("========================================================\n")
            for i, a in enumerate(fold_acc, 1):
                f.write(f"Fold {i} Accuracy: {a:.4f}\n")
            f.write(f"\nMean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n")
        print(f"Saved mean feature importances + summary to {output_dir}")

    lab_pred = lab_proba = None
    if lab_proba_sum is not None:
        lab_proba = lab_proba_sum / n_folds
        lab_pred = np.argmax(lab_proba, axis=1)

    return {"y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
            "fold_acc": fold_acc, "lab_pred": lab_pred, "lab_proba": lab_proba}


def main():
    """Standalone RF run (e.g. for feature-importance analysis).

    Pulls data through run_pipeline so the RF gets the EXACT same preprocessing
    as the CNN -- the single source of truth is run_pipeline.PREPROCESS_CONFIG.
    """
    # Imported here (not at module top) to avoid a circular import: run_pipeline
    # imports run_rf_cv from this module at load time.
    from run_pipeline import extract_data, preprocess_data, PREPROCESS_CONFIG, OUTPUT_DIR

    X, y, wn, _, _ = extract_data(include_lab=False)
    Xp, _ = preprocess_data(X, wn, PREPROCESS_CONFIG)
    run_rf_cv(Xp, y, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
