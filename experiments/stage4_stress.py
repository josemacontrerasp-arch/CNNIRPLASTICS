"""Stage 4 -- weathered + OOD stress test + reject-option threshold.

Run (from project root, inside CNN_env; needs internet for the NIST probes):
    python -m experiments.stage4_stress

Reuses the headline models saved by Stage 3 (no_os / norm-snv): the CNN fold
models are loaded from disk; the RF is cheap so it is retrained here. Answers:

  * Weathered plastics  -- FLOPP-e per-material behaviour (in-dist + OOD).
  * Bioplastic FPs      -- BLoP: how many bioplastics are forced into a
                           commodity class, and how confidently.
  * Chosen materials    -- NIST probes: stearic acid, ethylbenzene, ethyl
                           acetate, polyisobutylene/Vistanex (deliberate
                           near-misses to specific classes).
  * Reject option       -- confidence-threshold sweep: ROC of known (in-dist)
                           vs unknown (OOD) max-softmax, the optimal threshold
                           for CNN and RF, and the in-dist cost of rejecting.
  * Alternative to a    -- an explicit OTHER class (RF trained on six + a
    threshold              bioplastic OTHER class from OpenSpecy) and whether it
                           routes external bioplastics/NIST to OTHER.

Outputs: experiments_output/{results,tables,figures}/stage4_*.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from preprocess import preprocess
from experiments import config as C
from experiments import data as D
from experiments import external as E
from experiments import nist as NI
from experiments import models as M

N = len(C.SIX)
HEADLINE = "no_os__norm-snv"
HEADLINE_ARM = "no_os"
HEADLINE_CFG = "norm-snv"


def _prep(X, wn, cfg):
    Xp = preprocess(X, cfg, wavenumbers=wn)
    return np.asarray(Xp[0] if isinstance(Xp, tuple) else Xp, dtype=np.float32)


def _load_cnn_models():
    import tensorflow as tf  # noqa
    from tensorflow import keras
    paths = sorted((C.MODEL_DIR / HEADLINE).glob("cnn_fold_*.keras"))
    if not paths:
        raise FileNotFoundError(
            f"No CNN fold models in {C.MODEL_DIR / HEADLINE}. Run Stage 3 first.")
    return [keras.models.load_model(p) for p in paths]


def _maxprob(P):
    return P[np.arange(len(P)), np.argmax(P, axis=1)]


# --------------------------------------------------------------------------- #
def main():
    cfg = dict(C.sweep_configs())[HEADLINE_CFG]
    a = D.assemble(HEADLINE_ARM == "with_os", use_cache=True)
    X, y, names = D.build_xy(a, scope="six")
    wn = a["wn"]
    Xp = _prep(X, wn, cfg)

    # ---- known in-distribution scores from Stage 3 OOF --------------------
    oof = np.load(C.MODEL_DIR / HEADLINE / "oof.npz")
    known = {
        "cnn": dict(maxprob=_maxprob(oof["cnn_proba"]),
                    correct=(np.argmax(oof["cnn_proba"], 1) == oof["y_true"])),
        "rf": dict(maxprob=_maxprob(oof["rf_proba"]),
                   correct=(np.argmax(oof["rf_proba"], 1) == oof["rf_y_true"])),
    }

    # ---- models: load CNN, retrain RF ------------------------------------
    print("Loading headline CNN fold models + retraining RF...")
    cnn_models = _load_cnn_models()
    rf = M.rf_cv(Xp, y, N)
    proba_fns = {
        "cnn": lambda Xe: M.cnn_ensemble_proba(cnn_models, Xe),
        "rf": lambda Xe: M.rf_ensemble_proba(rf["forests"], Xe, N),
    }

    # ---- held-out sets ----------------------------------------------------
    lab = E.load_lab(wn)
    floppe = E.load_floppe(wn)
    blop = E.load_blop(wn)
    nist = NI.load_nist_probes(wn)
    nist_ok = [r for r in nist if "x" in r]
    for r in nist:
        if "error" in r:
            print(f"  NIST fetch FAILED for {r['file']}: {r['error']}")
    for rows in (lab, floppe, blop, nist_ok):
        for r in rows:
            r["x_pp"] = _prep(np.array([r["x"]]), wn, cfg)[0]

    floppe_id = [r for r in floppe if not r["ood"]]
    floppe_ood = [r for r in floppe if r["ood"]]
    ood_rows = blop + floppe_ood + nist_ok           # all unknown-to-the-6-class
    print(f"  known(OOF)={len(known['cnn']['maxprob'])}  "
          f"OOD set={len(ood_rows)} (BLoP {len(blop)} + FLOPP-e OOD "
          f"{len(floppe_ood)} + NIST {len(nist_ok)})")

    results = dict(headline=HEADLINE, n_known=int(len(known['cnn']['maxprob'])),
                   n_ood=len(ood_rows), models={})

    for mname in ("cnn", "rf"):
        pf = proba_fns[mname]
        # OOD predictions + confidences
        Xood = np.array([r["x_pp"] for r in ood_rows])
        Pood = pf(Xood)
        ood_conf = _maxprob(Pood)
        ood_pred = [C.INT_TO_SIX[i] for i in np.argmax(Pood, axis=1)]

        # ROC: known vs unknown using max-softmax (known = positive)
        yscore = np.concatenate([known[mname]["maxprob"], ood_conf])
        ylab = np.concatenate([np.ones(len(known[mname]["maxprob"])),
                               np.zeros(len(ood_conf))])
        auroc = float(skm.roc_auc_score(ylab, yscore))
        fpr, tpr, thr = skm.roc_curve(ylab, yscore)   # tpr=known kept, fpr=ood kept

        # threshold choices
        youden = thr[np.argmax(tpr - fpr)]
        # threshold that rejects >=95% of OOD (fpr<=0.05), maximizing known kept
        ok = np.where(fpr <= 0.05)[0]
        t95 = float(thr[ok[np.argmax(tpr[ok])]]) if len(ok) else float("nan")
        # in-dist cost at t95: fraction of known kept, and accuracy of kept-known
        kmp = known[mname]["maxprob"]
        kc = known[mname]["correct"]
        keep = kmp >= t95
        kept_frac = float(keep.mean())
        kept_acc = float(kc[keep].mean()) if keep.any() else float("nan")
        ood_rej = float((ood_conf < t95).mean())

        results["models"][mname] = dict(
            auroc_known_vs_ood=auroc,
            youden_threshold=float(youden),
            t_reject95=t95,
            known_kept_at_t95=kept_frac,
            known_acc_kept_at_t95=kept_acc,
            ood_rejected_at_t95=ood_rej,
            ood_mean_conf=float(ood_conf.mean()),
            ood_forced_ge_0p9=int((ood_conf >= 0.9).sum()),
            ood_n=len(ood_rows),
            roc=dict(fpr=fpr.tolist(), tpr=tpr.tolist(), thr=thr.tolist()),
        )
        print(f"  [{mname.upper()}] AUROC(known vs OOD)={auroc:.3f}  "
              f"t95={t95:.3f}  known kept {kept_frac*100:.0f}% "
              f"(acc {kept_acc*100:.1f}%)  OOD>=0.9: "
              f"{results['models'][mname]['ood_forced_ge_0p9']}/{len(ood_rows)}")

        # NIST probe table rows
        results["models"][mname]["nist"] = _probe_table(pf, nist_ok)
        # BLoP bioplastic FP rows
        results["models"][mname]["blop"] = _ood_material_table(pf, blop)
        # FLOPP-e OOD material table
        results["models"][mname]["floppe_ood"] = _ood_material_table(pf, floppe_ood)

    _plot_roc(results)
    _plot_threshold_curves(results, known, ood_rows, proba_fns)

    # ---- alternative: explicit OTHER class (RF on with_os six+other) ------
    results["other_class_experiment"] = _other_class_experiment(cfg)

    (C.RESULT_DIR / "stage4_stress.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_tables(results)
    print("\nStage 4 complete. See experiments_output/{results,tables,figures}/")


def _probe_table(pf, rows):
    if not rows:
        return []
    P = pf(np.array([r["x_pp"] for r in rows]))
    out = []
    for r, p in zip(rows, P):
        i = int(np.argmax(p))
        out.append(dict(name=r["file"], proxy_for=r.get("proxy_for", ""),
                        pred=C.INT_TO_SIX[i], conf=float(p[i]),
                        probs={C.INT_TO_SIX[j]: float(p[j]) for j in range(N)}))
    return out


def _ood_material_table(pf, rows):
    """Group OOD rows by material; report most-common pred + mean/max conf."""
    if not rows:
        return []
    P = pf(np.array([r["x_pp"] for r in rows]))
    preds = [C.INT_TO_SIX[i] for i in np.argmax(P, axis=1)]
    conf = _maxprob(P)
    by = {}
    for r, pr, c in zip(rows, preds, conf):
        by.setdefault(r["material"], []).append((pr, float(c)))
    out = []
    for mat, lst in sorted(by.items()):
        ps = [p for p, _ in lst]
        cs = [c for _, c in lst]
        common = max(set(ps), key=ps.count)
        out.append(dict(material=mat, n=len(lst), most_common_pred=common,
                        mean_conf=float(np.mean(cs)), max_conf=float(np.max(cs)),
                        n_ge_0p9=int(sum(c >= 0.9 for c in cs))))
    return out


def _other_class_experiment(headline_cfg):
    """Train RF on six + an OTHER class (OpenSpecy bioplastics) and test whether
    external bioplastics (BLoP) + NIST probes get routed to OTHER.

    Needs the with_os arm (bioplastic training spectra only exist in OpenSpecy).
    This is the 'reject option as an explicit class' alternative to thresholding.
    """
    cfg = dict(C.sweep_configs())["norm-snv"]
    a = D.assemble(True, use_cache=True)
    X, y, names = D.build_xy(a, scope="six+other")
    wn = a["wn"]
    Xp = _prep(X, wn, cfg)
    n_classes = len(names)
    other_idx = names.index("OTHER")

    rf = M.rf_cv(Xp, y, n_classes)
    # external OOD aligned to with_os grid
    blop = E.load_blop(wn)
    nist = [r for r in NI.load_nist_probes(wn) if "x" in r]
    for rows in (blop, nist):
        for r in rows:
            r["x_pp"] = _prep(np.array([r["x"]]), wn, cfg)[0]

    def route(rows):
        if not rows:
            return dict(n=0, to_other=0, frac_other=float("nan"))
        P = M.rf_ensemble_proba(rf["forests"], np.array([r["x_pp"] for r in rows]),
                                n_classes)
        pred = np.argmax(P, axis=1)
        to_other = int((pred == other_idx).sum())
        return dict(n=len(rows), to_other=to_other,
                    frac_other=float(to_other / len(rows)))

    from experiments import metrics as MET
    sc = MET.per_class_scores(rf["y_true"], rf["y_pred"], names)
    return dict(
        scope="six+other (OpenSpecy bioplastics as OTHER)",
        cv_macro_f1=sc["_overall"]["macro_f1"],
        other_class=sc.get("OTHER"),
        blop_routing=route(blop),
        nist_routing=route(nist),
        note="frac_other = share of external OOD correctly sent to OTHER; "
             "compare to the threshold reject rate in models.rf.ood_rejected_at_t95",
    )


# --------------------------------------------------------------------------- #
def _plot_roc(results):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for mname in ("cnn", "rf"):
        m = results["models"][mname]
        ax.plot(m["roc"]["fpr"], m["roc"]["tpr"],
                label=f"{mname.upper()} (AUROC={m['auroc_known_vs_ood']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("OOD kept (false accept rate)")
    ax.set_ylabel("Known kept (true accept rate)")
    ax.set_title("Reject option: known vs OOD by max-softmax")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "stage4_roc.png", dpi=150)
    plt.close(fig)


def _plot_threshold_curves(results, known, ood_rows, proba_fns):
    ts = np.linspace(0.3, 1.0, 71)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, mname in zip(axes, ("cnn", "rf")):
        pf = proba_fns[mname]
        Pood = pf(np.array([r["x_pp"] for r in ood_rows]))
        oc = Pood[np.arange(len(Pood)), np.argmax(Pood, 1)]
        kmp = known[mname]["maxprob"]
        kc = known[mname]["correct"]
        keep_frac = [float((kmp >= t).mean()) for t in ts]
        kept_acc = [float(kc[kmp >= t].mean()) if (kmp >= t).any() else np.nan for t in ts]
        ood_rej = [float((oc < t).mean()) for t in ts]
        ax.plot(ts, keep_frac, label="known kept")
        ax.plot(ts, kept_acc, label="acc of kept known")
        ax.plot(ts, ood_rej, label="OOD rejected")
        ax.axvline(results["models"][mname]["t_reject95"], color="r", ls="--",
                   alpha=0.6, label="t@95% OOD reject")
        ax.set_title(f"{mname.upper()} threshold trade-off")
        ax.set_xlabel("confidence threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "stage4_threshold_curves.png", dpi=150)
    plt.close(fig)


def _write_tables(results):
    # reject-option summary
    md = ["# Stage 4 -- reject-option threshold (headline model)\n",
          "| Model | AUROC known/OOD | Youden t | t@95% OOD-reject | "
          "known kept @t | acc of kept | OOD forced >=0.9 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for mname in ("cnn", "rf"):
        m = results["models"][mname]
        md.append(f"| {mname.upper()} | {m['auroc_known_vs_ood']:.3f} | "
                  f"{m['youden_threshold']:.3f} | {m['t_reject95']:.3f} | "
                  f"{m['known_kept_at_t95']*100:.0f}% | "
                  f"{m['known_acc_kept_at_t95']*100:.1f}% | "
                  f"{m['ood_forced_ge_0p9']}/{m['ood_n']} |")
    oc = results["other_class_experiment"]
    md += ["\n## Alternative: explicit OTHER class (RF, OpenSpecy bioplastics)\n",
           f"- CV macro-F1 (7-class): {oc['cv_macro_f1']:.3f}",
           f"- BLoP routed to OTHER: {oc['blop_routing']['to_other']}/"
           f"{oc['blop_routing']['n']} "
           f"({oc['blop_routing']['frac_other']*100:.0f}%)",
           f"- NIST probes routed to OTHER: {oc['nist_routing']['to_other']}/"
           f"{oc['nist_routing']['n']}",
           f"- {oc['note']}"]
    (C.TABLE_DIR / "stage4_reject.md").write_text("\n".join(md), encoding="utf-8")

    # NIST probe table (both models)
    md = ["# Stage 4 -- NIST near-miss probes (all OOD; high confidence = bad)\n",
          "| Probe | proxy for | CNN pred | CNN conf | RF pred | RF conf |",
          "|---|---|---|---:|---|---:|"]
    cnn_n = {r["name"]: r for r in results["models"]["cnn"]["nist"]}
    rf_n = {r["name"]: r for r in results["models"]["rf"]["nist"]}
    for name in cnn_n:
        c, r = cnn_n[name], rf_n.get(name, {})
        md.append(f"| {name} | {c['proxy_for']} | {c['pred']} | {c['conf']*100:.1f}% | "
                  f"{r.get('pred','-')} | {r.get('conf',float('nan'))*100:.1f}% |")
    (C.TABLE_DIR / "stage4_nist_probes.md").write_text("\n".join(md), encoding="utf-8")

    # bioplastic FP table (CNN + RF)
    for mname in ("cnn", "rf"):
        rows = results["models"][mname]["blop"]
        md = [f"# Stage 4 -- BLoP bioplastic false positives ({mname.upper()})\n",
              "| Bioplastic | n | most-common pred | mean conf | max conf | n>=0.9 |",
              "|---|---:|---|---:|---:|---:|"]
        for r in rows:
            md.append(f"| {r['material']} | {r['n']} | {r['most_common_pred']} | "
                      f"{r['mean_conf']*100:.0f}% | {r['max_conf']*100:.0f}% | "
                      f"{r['n_ge_0p9']} |")
        (C.TABLE_DIR / f"stage4_blop_{mname}.md").write_text("\n".join(md), encoding="utf-8")

    # FLOPP-e OOD material table (CNN)
    rows = results["models"]["cnn"]["floppe_ood"]
    md = ["# Stage 4 -- FLOPP-e OOD materials (CNN; forced into 6 classes)\n",
          "| Material | n | most-common pred | mean conf | max conf | n>=0.9 |",
          "|---|---:|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['material']} | {r['n']} | {r['most_common_pred']} | "
                  f"{r['mean_conf']*100:.0f}% | {r['max_conf']*100:.0f}% | {r['n_ge_0p9']} |")
    (C.TABLE_DIR / "stage4_floppe_ood_cnn.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
