# Experiment harness (1D-CNN + Random Forest study)

Orchestration layer on top of the existing modules (`data/format_data.py`,
`preprocess.py`, `models/cnn_model_draft.py`, `models/rf_model.py`). It answers
the project questions and writes every deliverable to `experiments_output/`.

We are building and running this **stage by stage** (incremental loop): run a
stage in `CNN_env`, send the console output / new files back, then the next
stage is built/tuned from the real numbers. The 2D CNN is out of scope.

## Run order (from the project root, inside CNN_env)

```bash
python -m experiments.stage1_summary      # assemble + cache data, dataset summary   [READY]
python -m experiments.stage2_rf_sweep      # RF preprocessing sweep + OpenSpecy A/B    [next]
python -m experiments.stage3_headtohead    # CNN vs RF on the chosen config            [next]
python -m experiments.stage4_stress        # FLOPP-e + bioplastic OOD + reject option  [next]
```

## What each stage answers

| Stage | Question(s) | Key outputs |
|---|---|---|
| 1 | Data sanity, OpenSpecy grid-collapse | class x source counts, grid stats, class-distribution chart |
| 2 | Best preprocessing config? Does OpenSpecy help/hurt? | RF sweep table (CV + FLOPP-e), A/B report |
| 3 | CNN vs RF on the winner | confusion matrices, per-class bars, FP/FN tables, lab generalization |
| 4 | Weathered (FLOPP-e), bioplastic FPs, reject threshold | per-material tables, OOD ROC, threshold recommendation |

## Outputs

Everything lands under `experiments_output/`:
`cache/` (aligned .npy), `figures/`, `tables/`, `models/`, `results/` (json).
Nothing in the pre-existing `output/` or `results/` is touched.

## Notes / decisions

* The regenerated OpenSpecy now contains PLA/PHA/PBS/PBAT. These are **not**
  target classes; the harness filters them out of the six-class task (and can
  optionally pool them into an `OTHER` class for the reject-option experiment).
  This sidesteps a crash in `run_pipeline.extract_data`, which raises on the
  unmapped PBS/PBAT labels.
* Preprocessing is swept cheaply with RF, then only the top configs go to the
  (expensive) CNN.
