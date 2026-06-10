"""Training wrappers that return a uniform result dict for both RF and CNN.

Both produce out-of-fold (OOF) predictions over the training data -- every
spectrum predicted exactly once by a fold that did not see it -- plus an
ensemble predict function for scoring held-out external/lab spectra. Keeping the
output shape identical means metrics.py scores both the same way.
"""
from __future__ import annotations

import numpy as np
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from experiments import config as C


# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
def rf_cv(X, y, n_classes, n_estimators=200, class_weight=None,
          n_folds=C.N_FOLDS, seed=C.SEED, keep_forests=True):
    """Stratified k-fold RF. Returns OOF preds/proba, per-fold acc, and forests.

    `class_weight=None` (default) or "balanced" to counter the OpenSpecy class
    imbalance -- used in the A/B to separate the imbalance effect from the
    grid/instrument effect.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_acc, forests = [], []
    proba = np.zeros((len(y), n_classes))   # indexed in shuffled-position space

    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                    n_jobs=-1, class_weight=class_weight)
        rf.fit(X[tr], y[tr])
        p = np.zeros((len(te), n_classes))
        p[:, rf.classes_] = rf.predict_proba(X[te])
        proba[te] = p
        fold_acc.append(skm.accuracy_score(y[te], np.argmax(p, axis=1)))
        if keep_forests:
            forests.append(rf)

    # Everything below is in the SAME shuffled-position order, so y_true, y_pred,
    # and proba correspond row-for-row (needed for the threshold/OOD analysis).
    y_pred = np.argmax(proba, axis=1)
    return dict(y_true=y, y_pred=y_pred, proba=proba, fold_acc=fold_acc,
                forests=forests, mean_acc=float(np.mean(fold_acc)),
                std_acc=float(np.std(fold_acc)))


def rf_ensemble_proba(forests, X_ext, n_classes):
    """Average predict_proba across fold forests for held-out spectra."""
    X_ext = np.asarray(X_ext)
    acc = np.zeros((len(X_ext), n_classes))
    for rf in forests:
        p = np.zeros((len(X_ext), n_classes))
        p[:, rf.classes_] = rf.predict_proba(X_ext)
        acc += p
    return acc / len(forests)


# --------------------------------------------------------------------------- #
# 1-D CNN (imported lazily so RF-only stages don't need TensorFlow)
# --------------------------------------------------------------------------- #
def cnn_cv(X, y, n_classes, n_folds=C.N_FOLDS, seed=C.SEED,
           dropout=0.2, learning_rate=1e-4, batch_size=64,
           epochs=1000, patience=100, save_dir=None, verbose=0):
    """Stratified k-fold 1-D CNN. Mirrors rf_cv's return shape.

    dropout / learning_rate / batch_size are exposed so stage 3 can tune them
    (the brief explicitly allows this, esp. for robustness). Heavy: GPU advised.
    """
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    from tensorflow.keras import layers

    X = np.asarray(X)[..., np.newaxis]
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    def build():
        np.random.seed(seed)
        tf.random.set_seed(seed)
        inp = layers.Input((X.shape[1], 1))
        h = layers.Conv1D(64, 3, activation="relu")(inp)
        h = layers.Conv1D(64, 3, activation="relu")(h)
        h = layers.MaxPool1D()(h)
        h = layers.Conv1D(64, 3, activation="relu")(h)
        h = layers.Conv1D(64, 3, activation="relu")(h)
        h = layers.MaxPool1D()(h)
        h = layers.Flatten()(h)
        for _ in range(3):
            h = layers.Dense(64, activation="relu")(h)
            h = layers.Dropout(dropout)(h)
        out = layers.Dense(n_classes, activation="softmax")(h)
        m = keras.Model(inp, out)
        m.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["acc"])
        return m

    from sklearn.model_selection import StratifiedKFold, train_test_split
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_acc, models = [], []
    proba = np.zeros((len(y), n_classes))   # shuffled-position space

    for k, (tr, te) in enumerate(skf.split(X[:, :, 0], y), 1):
        m = build()
        Xtr, Xval, ytr, yval = train_test_split(X[tr], y[tr], test_size=0.3,
                                                random_state=seed)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                           mode="min", restore_best_weights=True)
        m.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=epochs,
              batch_size=batch_size, shuffle=True, verbose=verbose, callbacks=[es])
        proba[te] = m.predict(X[te], verbose=0)
        fold_acc.append(skm.accuracy_score(y[te], np.argmax(proba[te], axis=1)))
        models.append(m)
        if save_dir is not None:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            m.save(Path(save_dir) / f"cnn_fold_{k}.keras")

    y_pred = np.argmax(proba, axis=1)
    return dict(y_true=y, y_pred=y_pred, proba=proba, fold_acc=fold_acc,
                models=models, mean_acc=float(np.mean(fold_acc)),
                std_acc=float(np.std(fold_acc)))


def cnn_ensemble_proba(models, X_ext):
    X_ext = np.asarray(X_ext)[..., np.newaxis]
    return np.mean([m.predict(X_ext, verbose=0) for m in models], axis=0)
