"""Stage 3 -- CNN vs RF head-to-head on the chosen config/data.

Run (from project root, inside CNN_env; GPU strongly advised for the CNN):
    python -m experiments.stage3_headtohead

Decisions carried in from Stage 2:
  * Shipping/headline data = c4+c8 only (no_os). OpenSpecy gave no transfer gain
    and worse in-domain CV, so it is demoted to OOD test material. We still train
    ONE CNN on with_os/snv to confirm the verdict holds for the CNN, not just RF.
  * Preprocessing = per-spectrum SNV headline, with smooth+d1+snv as a CNN
    tie-break (the RF sweep was saturated and couldn't separate configs).

RUNS (arm, config):
  ("no_os",  "norm-snv")        -> headline head-to-head
  ("no_os",  "smooth+d1+snv")   -> does a derivative help the CNN?
  ("with_os","norm-snv")        -> OpenSpecy confirmation for the CNN

For every run and BOTH models we emit the brief's deliverables:
  * confusion matrix (in-domain OOF)
  * per-class accuracy/precision/recall/F1 bar chart
  * false-positive / false-negative tables
  * lab generalization (39 held-out real-world spectra) + confusion
  * FLOPP-e in-distribution accuracy
  * BLoP bioplastic over-confidence count (>=90%) -- the CNN-vs-RF safety gap
Fold models + OOF/external probabilities are saved for Stage 4 to reuse.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import preprocess
from experiments import config as C
from experiments import data as D
from experiments import external as E
from experiments import models as M
from experiments import metrics as MET

N_CLASSES = len(C.SIX)

RUNS = [("no_os", "norm-snv"), ("no_os", "smooth+d1+snv"), ("with_os", "norm-snv")]
HEADLINE = ("no_os", "norm-snv")

# CNN hyperparameters (tunable per the brief). batch_size=128 for GPU throughput
# (Tesla T4 has headroom); epochs/patience trimmed from the original 1000/100
# because the data is easily separable, saving a lot of time at ~no accuracy cost.
CNN_KW = dict(dropout=0.2, learning_rate=1e-4, batch_size=128,
              epochs=300, patience=30, verbose=0)


def _prep(X, wn, cfg):
    Xp = preprocess(X, cfg, wavenumbers=wn)
    return np.asarray(Xp[0] if isinstance(Xp, tuple) else Xp, dtype=np.float32)


def _eval_set(proba_fn, rows, class_names):
    """Run an ensemble on external/lab rows. Returns pred labels, confs, proba."""
    if not rows:
        return [], np.array([]), None
    X = np.array([r["x_pp"] for r in rows], float)
    P = proba_fn(X)
    idx = np.argmax(P, axis=1)
    pred = [class_names[i] for i in idx]
    conf = P[np.arange(len(P)), idx]
    return pred, conf, P


def _run_one(arm, cfgname, fig_root):
    cfg = dict(C.sweep_configs())[cfgname]
    a = D.assemble(arm == "with_os", use_cache=True)
    X, y, names = D.build_xy(a, scope="six")
    wn = a["wn"]
    Xp = _prep(X, wn, cfg)

    # held-out evaluation sets, aligned to THIS grid + same preprocessing
    lab = E.load_lab(wn)
    floppe = E.load_floppe(wn)
    blop = E.load_blop(wn)
    for rows in (lab, floppe, blop):
        for r in rows:
            r["x_pp"] = _prep(np.array([r["x"]]), wn, cfg)[0]

    runtag = f"{arm}__{cfgname}"
    fig_dir = fig_root / runtag
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir = C.MODEL_DIR / runtag
    out = dict(arm=arm, config=cfgname, n_train=int(len(y)),
               n_points=int(Xp.shape[1]))

    # ---- train both models -------------------------------------------------
    print(f"\n=== {runtag}: RF ===")
    rf = M.rf_cv(Xp, y, N_CLASSES)
    print(f"    RF CV acc {rf['mean_acc']:.4f} (+/-{rf['std_acc']:.4f})")
    print(f"=== {runtag}: CNN ===")
    cnn = M.cnn_cv(Xp, y, N_CLASSES, save_dir=model_dir, **CNN_KW)
    print(f"    CNN CV acc {cnn['mean_acc']:.4f} (+/-{cnn['std_acc']:.4f})")

    # save OOF probabilities for Stage 4
    np.savez(model_dir / "oof.npz", y_true=cnn["y_true"],
             cnn_proba=cnn["proba"], rf_y_true=rf["y_true"], rf_proba=rf["proba"])

    for mname, res, proba_fn in [
        ("rf", rf, lambda Xe: M.rf_ensemble_proba(rf["forests"], Xe, N_CLASSES)),
        ("cnn", cnn, lambda Xe: M.cnn_ensemble_proba(cnn["models"], Xe)),
    ]:
        sc = MET.per_class_scores(res["y_true"], res["y_pred"], names)
        MET.plot_confusion(res["y_true"], res["y_pred"], names,
                           f"{mname.upper()} {runtag} (OOF)  acc={sc['_overall']['accuracy']:.3f}",
                           fig_dir / f"confusion_{mname}.png")
        MET.plot_per_class_bars(sc, names, f"{mname.upper()} {runtag} per-class",
                                fig_dir / f"perclass_{mname}.png")
        MET.fp_fn_tables(res["y_true"], res["y_pred"], names,
                         C.TABLE_DIR / f"stage3_{runtag}_{mname}_fpfn")

        # lab generalization
        lab_pred, lab_conf, _ = _eval_set(proba_fn, lab, names)
        lab_ok = sum(p == r["truth"] or (r["truth"] == "PE" and p in C.PE_ACCEPT)
                     for p, r in zip(lab_pred, lab))
        lab_acc = lab_ok / len(lab) if lab else float("nan")
        if lab:
            yt = [C.SIX_TO_INT[r["truth"]] for r in lab]
            yp = [C.SIX_TO_INT[p] for p in lab_pred]
            MET.plot_confusion(yt, yp, names,
                               f"{mname.upper()} {runtag} LAB acc={lab_acc:.3f}",
                               fig_dir / f"confusion_{mname}_lab.png")

        # FLOPP-e in-dist + BLoP over-confidence
        fp, fc, _ = _eval_set(proba_fn, floppe, names)
        sidf = E.score_indist(floppe, fp, fc)
        bp, bc, _ = _eval_set(proba_fn, blop, names)
        blop_hi = int((bc >= 0.9).sum()) if len(bc) else 0

        out[mname] = dict(
            cv_acc=sc["_overall"]["accuracy"], cv_macro_f1=sc["_overall"]["macro_f1"],
            per_class={k: v for k, v in sc.items() if k != "_overall"},
            lab_acc=lab_acc, lab_n=len(lab),
            floppe_id_acc=sidf["id_acc"], floppe_ok=sidf["n_ok"], floppe_n=sidf["n_id"],
            blop_hi_conf=blop_hi, blop_n=len(blop),
            blop_mean_conf=float(np.mean(bc)) if len(bc) else float("nan"),
            fold_acc=res["fold_acc"])
        print(f"    {mname.upper()}: CV macroF1={out[mname]['cv_macro_f1']:.3f} "
              f"lab={lab_acc*100:.1f}% FLOPP-e={sidf['id_acc']*100:.1f}% "
              f"BLoP>=90%={blop_hi}/{len(blop)}")

    return runtag, out


def main():
    fig_root = C.FIG_DIR / "stage3"
    results = {}
    for arm, cfgname in RUNS:
        tag, out = _run_one(arm, cfgname, fig_root)
        results[tag] = out

    (C.RESULT_DIR / "stage3_headtohead.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_summary(results)
    print("\nStage 3 complete. See experiments_output/{results,tables,figures/stage3,models}/")


def _write_summary(results):
    md = ["# Stage 3 -- CNN vs RF head-to-head\n",
          "CV = in-domain held-out folds. Lab/FLOPP-e = held-out generalization. "
          "BLoP>=90% = bioplastics forced into a commodity class with >=90% "
          "confidence (the CNN-vs-RF safety gap; lower is better).\n",
          "| Run | Model | CV macro-F1 | Lab acc | FLOPP-e acc | BLoP forced >=90% |",
          "|---|---|---:|---:|---:|---:|"]
    for tag, out in results.items():
        for m in ("cnn", "rf"):
            if m not in out:
                continue
            r = out[m]
            md.append(f"| {tag} | {m.upper()} | {r['cv_macro_f1']:.3f} | "
                      f"{r['lab_acc']*100:.1f}% ({r['lab_n']}) | "
                      f"{r['floppe_id_acc']*100:.1f}% ({r['floppe_ok']}/{r['floppe_n']}) | "
                      f"{r['blop_hi_conf']}/{r['blop_n']} |")
    (C.TABLE_DIR / "stage3_headtohead.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
