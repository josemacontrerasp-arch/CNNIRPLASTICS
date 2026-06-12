"""Model-agnostic scoring + plotting.

Everything consumes integer-label predictions (y_true, y_pred) and optional
probabilities, exactly the format the CNN's test_model and the RF emit, so the
CNN, the RF, and any future model are scored identically.

Outputs the deliverables the brief asks for:
  * confusion matrices (PNG)
  * per-class bar charts of accuracy / precision / recall / F1
  * false-positive and false-negative tables (per class, CSV + markdown)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn.metrics as skm


def per_class_scores(y_true, y_pred, class_names):
    """Return a dict of per-class accuracy/precision/recall/F1 + macro/overall.

    Per-class "accuracy" here is one-vs-rest accuracy (correctly placed in or
    out of the class), which is the intuitive bar most people expect alongside
    precision/recall/F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))

    prec = skm.precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = skm.recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = skm.f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # one-vs-rest accuracy per class
    ovr_acc = []
    for i in labels:
        t = (y_true == i)
        p = (y_pred == i)
        ovr_acc.append(float(np.mean(t == p)))

    out = {}
    for i, name in enumerate(class_names):
        out[name] = dict(accuracy=float(ovr_acc[i]), precision=float(prec[i]),
                         recall=float(rec[i]), f1=float(f1[i]),
                         support=int(np.sum(y_true == i)))
    out["_overall"] = dict(
        accuracy=float(skm.accuracy_score(y_true, y_pred)),
        macro_f1=float(skm.f1_score(y_true, y_pred, labels=labels,
                                    average="macro", zero_division=0)),
        weighted_f1=float(skm.f1_score(y_true, y_pred, labels=labels,
                                       average="weighted", zero_division=0)),
    )
    return out


def plot_confusion(y_true, y_pred, class_names, title, path, normalize=False):
    cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = cm.astype(float)
    if normalize:
        rs = disp.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        disp = disp / rs

    fig, ax = plt.subplots(figsize=(0.7 * len(class_names) + 2.5,
                                    0.7 * len(class_names) + 2.0))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thr = disp.max() / 2 if disp.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{disp[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_per_class_bars(scores: dict, class_names, title, path):
    """Grouped bar chart: accuracy/precision/recall/F1 for each class."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(class_names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(class_names)), 4.2))
    for k, m in enumerate(metrics):
        vals = [scores[c][m] for c in class_names]
        ax.bar(x + (k - 1.5) * w, vals, w, label=m)
    ax.set_xticks(x, class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title, fontsize=10)
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fp_fn_tables(y_true, y_pred, class_names, out_prefix: Path, extra=None):
    """Write per-class false-positive and false-negative breakdowns.

    A false positive for class C = a sample predicted C whose truth != C.
    A false negative for class C = a sample whose truth is C but predicted != C.
    We tabulate the confusion pairs so it is clear WHICH class each FP/FN leaked
    to/from. `extra` (optional list of dicts, len == n samples) can carry source
    or filename info for per-sample drill-down.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for ci, c in enumerate(class_names):
        fp_mask = (y_pred == ci) & (y_true != ci)
        fn_mask = (y_true == ci) & (y_pred != ci)
        # which classes the FPs actually came from
        fp_from = {}
        for t in y_true[fp_mask]:
            fp_from[class_names[t]] = fp_from.get(class_names[t], 0) + 1
        fn_to = {}
        for p in y_pred[fn_mask]:
            fn_to[class_names[p]] = fn_to.get(class_names[p], 0) + 1
        rows.append(dict(
            cls=c,
            support=int(np.sum(y_true == ci)),
            false_positives=int(fp_mask.sum()),
            fp_sources=fp_from,
            false_negatives=int(fn_mask.sum()),
            fn_leaks_to=fn_to,
        ))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "false_positives", "fp_sources",
                    "false_negatives", "fn_leaks_to"])
        for r in rows:
            w.writerow([r["cls"], r["support"], r["false_positives"],
                        json.dumps(r["fp_sources"]), r["false_negatives"],
                        json.dumps(r["fn_leaks_to"])])

    md = ["| Class | Support | FPs | FP came from | FNs | FN leaked to |",
          "|---|---:|---:|---|---:|---|"]
    for r in rows:
        fp_s = ", ".join(f"{k}:{v}" for k, v in r["fp_sources"].items()) or "-"
        fn_s = ", ".join(f"{k}:{v}" for k, v in r["fn_leaks_to"].items()) or "-"
        md.append(f"| {r['cls']} | {r['support']} | {r['false_positives']} | "
                  f"{fp_s} | {r['false_negatives']} | {fn_s} |")
    out_prefix.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return rows
