"""Persist the recommended final models so deployment needs no retraining.

Run (CPU is fine, fast):
    python -m experiments.save_final_models

Writes to experiments_output/models/final/:
  RANDOM FOREST (recommended deployment classifier + reject gate)
    * rf_final.pkl        -- single RF trained on ALL c4+c8 / SNV data
    * rf_fold_1..5.pkl    -- the 5 CV forests (the soft-vote ensemble scored in
                             the study)
  1-D CNN (the headline no_os/SNV ensemble -- the model behind every CNN number
  in the report: 99.83% CV, 92.3% lab)
    * cnn_fold_1..5.keras -- copied from models/no_os__norm-snv/ (the 5-fold
                             soft-vote ensemble IS the deployment CNN)
    * cnn_oof.npz         -- its out-of-fold probabilities
  * deployment.json       -- class order, preprocessing, grid, the 0.89 reject
                             threshold, and which model is which.

Why these and not the others: smooth+d1+snv hurt transfer and with_os (OpenSpecy)
hurt both CV and lab, so the no_os/SNV pair is the only one recommended for use.
The RF is the headline model (beat the CNN on lab transfer and is the only one
with a usable reject threshold); the CNN ensemble is persisted as the secondary
model for anyone who wants it, but its softmax should be temperature-scaled
before being used as a confidence.
"""
from __future__ import annotations

import json
import pickle
import shutil

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from preprocess import preprocess
from experiments import config as C
from experiments import data as D
from experiments import models as M

CFG_NAME = "norm-snv"
THRESHOLD = 0.89
HEADLINE_RUN = "no_os__norm-snv"


def main():
    cfg = dict(C.sweep_configs())[CFG_NAME]
    a = D.assemble(False, use_cache=True)          # no_os = c4 + c8
    X, y, names = D.build_xy(a, scope="six")
    wn = a["wn"]
    Xp = preprocess(X, cfg, wavenumbers=wn)
    Xp = np.asarray(Xp[0] if isinstance(Xp, tuple) else Xp, dtype=np.float32)

    out = C.MODEL_DIR / "final"
    out.mkdir(parents=True, exist_ok=True)

    # ---- Random Forest ----------------------------------------------------
    rf = M.rf_cv(Xp, y, len(names))
    for i, forest in enumerate(rf["forests"], 1):
        with open(out / f"rf_fold_{i}.pkl", "wb") as f:
            pickle.dump(forest, f)
    rf_full = RandomForestClassifier(n_estimators=200, random_state=C.SEED, n_jobs=-1)
    rf_full.fit(Xp, y)
    with open(out / "rf_final.pkl", "wb") as f:
        pickle.dump(rf_full, f)
    print(f"RF: saved 5 CV forests (mean CV acc {rf['mean_acc']:.4f}) + "
          f"rf_final.pkl (all {len(y)} spectra)")

    # ---- 1-D CNN (copy the headline 5-fold ensemble; no retraining) -------
    src = C.MODEL_DIR / HEADLINE_RUN
    keras_files = sorted(src.glob("cnn_fold_*.keras"))
    if not keras_files:
        print(f"  WARNING: no CNN folds in {src}. Run Stage 3 first, then re-run.")
        n_cnn = 0
    else:
        for i, kf in enumerate(keras_files, 1):
            shutil.copy2(kf, out / f"cnn_fold_{i}.keras")
        if (src / "oof.npz").exists():
            shutil.copy2(src / "oof.npz", out / "cnn_oof.npz")
        n_cnn = len(keras_files)
        print(f"CNN: copied {n_cnn} headline fold models ({HEADLINE_RUN}) -> final/")

    # ---- deployment metadata ---------------------------------------------
    meta = dict(
        classes=names,
        preprocessing=CFG_NAME,
        grid_points=int(Xp.shape[1]),
        grid_range_cm1=[float(wn.min()), float(wn.max())],
        recommended_model="random_forest",
        random_forest=dict(
            single="rf_final.pkl",
            ensemble=[f"rf_fold_{i}.pkl" for i in range(1, 6)],
            note="Soft-vote the 5 forests (or use rf_final for a single model). "
                 "Headline classifier + reject gate.",
        ),
        cnn=dict(
            ensemble=[f"cnn_fold_{i}.keras" for i in range(1, n_cnn + 1)],
            source_run=HEADLINE_RUN,
            note="5-fold soft-vote ensemble = the deployment CNN. Secondary "
                 "model; softmax is NOT calibrated for a reject option "
                 "(AUROC 0.836) -- temperature-scale before using as confidence.",
        ),
        reject_threshold=THRESHOLD,
        reject_rule=(f"if max RF class probability < {THRESHOLD}: predict "
                     "'UNKNOWN' (not one of the six). Keeps 97.6% of real "
                     "plastics (100% correct) and rejects 96.5% of OOD."),
    )
    (out / "deployment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved deployment.json -> {out / 'deployment.json'}")
    print("\nFinal models saved to experiments_output/models/final/ "
          f"(RF + {n_cnn} CNN folds).")


if __name__ == "__main__":
    main()
