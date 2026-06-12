"""Stage 2 -- RF preprocessing sweep + OpenSpecy A/B.

Run (from project root, inside CNN_env):
    python -m experiments.stage2_rf_sweep

Part A -- preprocessing sweep (on the clean, balanced no_os arm)
    Walk the curated config menu (config.sweep_configs). For each, RF 5-fold CV
    on the six commodity classes, scored by macro-F1, PVC recall (the at-risk
    class), and -- the real generalization signal -- FLOPP-e in-distribution
    accuracy. Random Forest is the cheap proxy that lets us rank configs without
    paying for a CNN each time.

Part B -- OpenSpecy A/B (does including OpenSpecy help or hurt?)
    Re-run the winning config (plus an SNV control) on BOTH arms:
        no_os   = FTIR c4+c8 (fine grid, balanced)
        with_os = + OpenSpecy (coarse grid, imbalanced)
    Report in-domain CV AND FLOPP-e transfer, per-class, with extra focus on
    PVC. For with_os we also run class_weight="balanced" to separate the
    imbalance effect from the grid/instrument effect.

Outputs: experiments_output/results/stage2_*.json, tables/*.md, figures/*.png.
RF only -- no TensorFlow needed.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import preprocess, PreprocessConfig
from experiments import config as C
from experiments import data as D
from experiments import external as E
from experiments import models as M
from experiments import metrics as MET

N_CLASSES = len(C.SIX)


# --------------------------------------------------------------------------- #
# Preprocessing with on-disk cache (baseline correction is the expensive part)
# --------------------------------------------------------------------------- #
def _prep_cached(X, wn, cfg, tag, cfgname):
    path = C.CACHE_DIR / f"Xp_{tag}_{cfgname}.npy"
    if path.exists():
        return np.load(path)
    Xp = preprocess(X, cfg, wavenumbers=wn)
    Xp = Xp[0] if isinstance(Xp, tuple) else Xp
    np.save(path, Xp.astype(np.float32))
    return Xp.astype(np.float32)


def _prep_external(rows, wn, cfg):
    """Preprocess a list of external rows with the same config + region trim."""
    if not rows:
        return np.empty((0, len(wn))), rows
    Xraw = np.array([r["x"] for r in rows], float)
    Xp = preprocess(Xraw, cfg, wavenumbers=wn)
    Xp = Xp[0] if isinstance(Xp, tuple) else Xp
    return np.asarray(Xp), rows


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #
def part_a():
    print("\n" + "#" * 72 + "\n# PART A -- preprocessing sweep (no_os arm)\n" + "#" * 72)
    a = D.assemble(False, use_cache=True)
    X, y, names = D.build_xy(a, scope="six")
    wn = a["wn"]
    floppe = E.load_floppe(wn)          # raw aligned to this grid
    print(f"  training: {len(y)} spectra x {X.shape[1]} pts | FLOPP-e: {len(floppe)} spectra")

    rows = []
    for cfgname, cfg in C.sweep_configs():
        Xp = _prep_cached(X, wn, cfg, "no_os", cfgname)
        Xpf, frows = _prep_external(floppe, wn, cfg)
        res = M.rf_cv(Xp, y, N_CLASSES)
        sc = MET.per_class_scores(res["y_true"], res["y_pred"], names)
        # FLOPP-e transfer
        P = M.rf_ensemble_proba(res["forests"], Xpf, N_CLASSES)
        pred = [C.INT_TO_SIX[i] for i in np.argmax(P, axis=1)]
        conf = P[np.arange(len(P)), np.argmax(P, axis=1)]
        sid = E.score_indist(frows, pred, conf)
        rec = dict(config=cfgname,
                   cv_acc=sc["_overall"]["accuracy"],
                   cv_macro_f1=sc["_overall"]["macro_f1"],
                   pvc_f1=sc["PVC"]["f1"], pvc_recall=sc["PVC"]["recall"],
                   hdpe_f1=sc["HDPE"]["f1"], ldpe_f1=sc["LDPE"]["f1"],
                   floppe_id_acc=sid["id_acc"], floppe_ok=sid["n_ok"],
                   floppe_n=sid["n_id"], n_points=int(Xp.shape[1]))
        rows.append(rec)
        print(f"  {cfgname:<18} CV macroF1={rec['cv_macro_f1']:.3f} "
              f"PVCrec={rec['pvc_recall']:.3f}  FLOPP-e={rec['floppe_id_acc']*100:5.1f}% "
              f"({rec['floppe_ok']}/{rec['floppe_n']})")

    rows.sort(key=lambda r: (round(r["floppe_id_acc"], 4), round(r["cv_macro_f1"], 4)),
              reverse=True)
    (C.RESULT_DIR / "stage2_sweep.json").write_text(json.dumps(rows, indent=2),
                                                    encoding="utf-8")
    _write_sweep_table(rows)
    _plot_sweep(rows)
    winner = rows[0]["config"]
    print(f"\n  >> winning config (FLOPP-e, then CV macro-F1): {winner}")
    return winner


def _write_sweep_table(rows):
    md = ["# Stage 2 Part A -- preprocessing sweep (RF, no_os arm)\n",
          "Ranked by FLOPP-e in-distribution accuracy, then CV macro-F1.\n",
          "| Config | CV acc | CV macro-F1 | PVC F1 | PVC recall | HDPE F1 | LDPE F1 | FLOPP-e acc |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['cv_acc']:.3f} | {r['cv_macro_f1']:.3f} | "
                  f"{r['pvc_f1']:.3f} | {r['pvc_recall']:.3f} | {r['hdpe_f1']:.3f} | "
                  f"{r['ldpe_f1']:.3f} | {r['floppe_id_acc']*100:.1f}% "
                  f"({r['floppe_ok']}/{r['floppe_n']}) |")
    (C.TABLE_DIR / "stage2_sweep.md").write_text("\n".join(md), encoding="utf-8")


def _plot_sweep(rows):
    names = [r["config"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(names)), 4.2))
    ax.bar(x - 0.2, [r["cv_macro_f1"] for r in rows], 0.4, label="CV macro-F1")
    ax.bar(x + 0.2, [r["floppe_id_acc"] for r in rows], 0.4, label="FLOPP-e acc")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Preprocessing sweep (RF, no_os): in-domain vs FLOPP-e transfer")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "stage2_sweep.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Part B -- OpenSpecy A/B
# --------------------------------------------------------------------------- #
def part_b(winner_cfgname):
    print("\n" + "#" * 72 + "\n# PART B -- OpenSpecy A/B\n" + "#" * 72)
    cfg_menu = dict(C.sweep_configs())
    test_cfgs = {"norm-snv": cfg_menu["norm-snv"]}
    if winner_cfgname in cfg_menu:
        test_cfgs[winner_cfgname] = cfg_menu[winner_cfgname]

    out = {}
    for include_os in (False, True):
        tag = "with_os" if include_os else "no_os"
        a = D.assemble(include_os, use_cache=True)
        X, y, names = D.build_xy(a, scope="six")
        wn = a["wn"]
        floppe = E.load_floppe(wn)
        blop = E.load_blop(wn)
        for cfgname, cfg in test_cfgs.items():
            weight_variants = [(None, "")]
            if include_os:
                weight_variants.append(("balanced", "_bal"))
            for cw, suffix in weight_variants:
                key = f"{tag}__{cfgname}{suffix}"
                Xp = _prep_cached(X, wn, cfg, tag, cfgname)
                res = M.rf_cv(Xp, y, N_CLASSES, class_weight=cw)
                sc = MET.per_class_scores(res["y_true"], res["y_pred"], names)
                Xpf, frows = _prep_external(floppe, wn, cfg)
                Pf = M.rf_ensemble_proba(res["forests"], Xpf, N_CLASSES)
                predf = [C.INT_TO_SIX[i] for i in np.argmax(Pf, axis=1)]
                conff = Pf[np.arange(len(Pf)), np.argmax(Pf, axis=1)]
                sidf = E.score_indist(frows, predf, conff)
                # bioplastic OOD: how confidently does it force BLoP into a class?
                Xpb, brows = _prep_external(blop, wn, cfg)
                Pb = M.rf_ensemble_proba(res["forests"], Xpb, N_CLASSES)
                confb = Pb[np.arange(len(Pb)), np.argmax(Pb, axis=1)]
                blop_hi = int((confb >= 0.9).sum())
                out[key] = dict(
                    arm=tag, config=cfgname, class_weight=cw,
                    n_train=int(len(y)), n_points=int(Xp.shape[1]),
                    cv_acc=sc["_overall"]["accuracy"],
                    cv_macro_f1=sc["_overall"]["macro_f1"],
                    pvc_f1=sc["PVC"]["f1"], pvc_recall=sc["PVC"]["recall"],
                    hdpe_recall=sc["HDPE"]["recall"], ldpe_recall=sc["LDPE"]["recall"],
                    floppe_id_acc=sidf["id_acc"], floppe_ok=sidf["n_ok"],
                    floppe_n=sidf["n_id"],
                    blop_hi_conf=blop_hi, blop_n=len(brows),
                    blop_mean_conf=float(np.mean(confb)) if len(confb) else float("nan"))
                print(f"  {key:<28} CVmF1={out[key]['cv_macro_f1']:.3f} "
                      f"PVCrec={out[key]['pvc_recall']:.3f} "
                      f"FLOPP-e={out[key]['floppe_id_acc']*100:5.1f}% "
                      f"BLoP>=90%={blop_hi}/{len(brows)}")

    (C.RESULT_DIR / "stage2_ab_openspecy.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    _write_ab_table(out)
    return out


def _write_ab_table(out):
    md = ["# Stage 2 Part B -- OpenSpecy A/B (RF)\n",
          "CV is in-domain (held-out folds). FLOPP-e is the transfer signal. "
          "BLoP>=90% counts bioplastics forced into a commodity class with high "
          "confidence (lower is better; the model should be uncertain on OOD).\n",
          "| Arm / config | n_train | pts | CV macro-F1 | PVC recall | "
          "HDPE rec | LDPE rec | FLOPP-e acc | BLoP forced >=90% |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in out.items():
        md.append(f"| {k} | {r['n_train']} | {r['n_points']} | {r['cv_macro_f1']:.3f} | "
                  f"{r['pvc_recall']:.3f} | {r['hdpe_recall']:.3f} | {r['ldpe_recall']:.3f} | "
                  f"{r['floppe_id_acc']*100:.1f}% ({r['floppe_ok']}/{r['floppe_n']}) | "
                  f"{r['blop_hi_conf']}/{r['blop_n']} |")
    (C.TABLE_DIR / "stage2_ab_openspecy.md").write_text("\n".join(md), encoding="utf-8")


def main():
    winner = part_a()
    part_b(winner)
    print("\nStage 2 complete. See experiments_output/{results,tables,figures}/")


if __name__ == "__main__":
    main()
