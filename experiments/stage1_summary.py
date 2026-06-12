"""Stage 1 -- assemble + cache the data and emit a dataset summary.

Run (from project root, inside CNN_env):
    python -m experiments.stage1_summary

What it does
------------
* Builds BOTH arms once and caches them to experiments_output/cache/:
    - no_os  : FTIR c4 + c8 only          (fine grid, ~1868 pts, 399-3999 cm^-1)
    - with_os: FTIR c4 + c8 + OpenSpecy    (coarse grid, ~394 pts, 804-3168 cm^-1)
* Prints class x source counts, grid range/step, and the OpenSpecy grid-collapse.
* Writes experiments_output/results/stage1_summary.json and a class-distribution
  bar chart, so we can confirm the data assembly + label handling before any
  modelling. This is also where the PBS/PBAT label fix is exercised.

Nothing is trained here. It is fast apart from the one-time CSV load+align.
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import config as C
from experiments import data as D


def _counts_by_class_source(assembled):
    labels = assembled["labels"]
    sources = assembled["sources"]
    table = {}
    for lab, src in zip(labels, sources):
        table.setdefault(lab, {}).setdefault(src, 0)
        table[lab][src] += 1
    return table


def _grid_info(assembled):
    wn = assembled["wn"]
    steps = np.abs(np.diff(wn))
    return dict(n_points=int(len(wn)), wn_min=float(wn.min()), wn_max=float(wn.max()),
                median_step=float(np.median(steps)))


def main():
    summary = {}
    for include_os in (False, True):
        tag = "with_os" if include_os else "no_os"
        print("\n" + "=" * 72)
        print(f"ARM: {tag}  (OpenSpecy {'INCLUDED' if include_os else 'EXCLUDED'})")
        print("=" * 72)
        a = D.assemble(include_os, use_cache=True)
        grid = _grid_info(a)
        print(f"  grid: {grid['n_points']} pts | {grid['wn_min']:.0f}-"
              f"{grid['wn_max']:.0f} cm^-1 | median step {grid['median_step']:.2f}")

        table = _counts_by_class_source(a)
        # print a compact class x source matrix
        sources = sorted({s for v in table.values() for s in v})
        print(f"  {'label':<8}" + "".join(f"{s:>12}" for s in sources) + f"{'TOTAL':>8}")
        scope_counts = {}
        for lab in sorted(table):
            row = table[lab]
            tot = sum(row.values())
            scope_counts[lab] = tot
            in_scope = "six" if lab in C.SIX_TO_INT else (
                "bio" if lab in C.BIOPLASTIC_LABELS else "??")
            print(f"  {lab:<8}" + "".join(f"{row.get(s, 0):>12}" for s in sources)
                  + f"{tot:>8}  [{in_scope}]")

        # six-class training matrix actually used
        Xs, y, names = D.build_xy(a, scope="six")
        six_counts = {names[i]: int((y == i).sum()) for i in range(len(names))}
        print(f"  six-class training spectra: {len(y)}  {six_counts}")

        lab_n = 0 if a["X_lab"] is None else len(a["X_lab"])
        print(f"  test-only lab spectra: {lab_n}")

        summary[tag] = dict(grid=grid, class_source_counts=table,
                            scope_counts=scope_counts, six_counts=six_counts,
                            n_six=int(len(y)), n_lab=lab_n)

    # --- grid-collapse headline -------------------------------------------
    g0, g1 = summary["no_os"]["grid"], summary["with_os"]["grid"]
    collapse = dict(
        points_no_os=g0["n_points"], points_with_os=g1["n_points"],
        pct_points_lost=round((1 - g1["n_points"] / g0["n_points"]) * 100, 1),
        lost_low_wn_region=[round(g0["wn_min"], 0), round(g1["wn_min"], 0)],
        step_no_os=g0["median_step"], step_with_os=g1["median_step"],
    )
    summary["grid_collapse"] = collapse
    print("\n" + "=" * 72)
    print("OPENSPECY GRID COLLAPSE")
    print(f"  points: {collapse['points_no_os']} -> {collapse['points_with_os']} "
          f"(-{collapse['pct_points_lost']}%)")
    print(f"  lost low-wn fingerprint region: "
          f"{collapse['lost_low_wn_region'][0]:.0f}-"
          f"{collapse['lost_low_wn_region'][1]:.0f} cm^-1")
    print(f"  step: {collapse['step_no_os']:.2f} -> {collapse['step_with_os']:.2f} cm^-1")

    (C.RESULT_DIR / "stage1_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {C.RESULT_DIR / 'stage1_summary.json'}")

    # --- class-distribution figure (six classes, per source, both arms) ----
    _plot_class_distribution(summary)


def _plot_class_distribution(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, tag in zip(axes, ("no_os", "with_os")):
        table = summary[tag]["class_source_counts"]
        classes = [c for c in C.SIX if c in table]
        sources = sorted({s for c in classes for s in table[c]})
        bottom = np.zeros(len(classes))
        for s in sources:
            vals = np.array([table[c].get(s, 0) for c in classes], float)
            ax.bar(classes, vals, bottom=bottom, label=s)
            bottom += vals
        ax.set_title(f"{tag}  (six commodity classes)", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("spectra")
    fig.suptitle("Class distribution by source", fontsize=11)
    fig.tight_layout()
    path = C.FIG_DIR / "stage1_class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
