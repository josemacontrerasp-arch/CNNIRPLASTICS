# CNN Predictions on IRSpirit FTIR Samples

Ensemble predictions from `output/fold_1.keras` ... `fold_4.keras` (soft-vote: average of the four softmax outputs, then argmax).

## Methodology

The new spectra were acquired on a Shimadzu IRSpirit and exported as `.txt`
(JCAMP-DX style: header lines + `wavenumber  %T` pairs, ~630 points,
399.85 - 3998.48 cm-1).

Preprocessing (mirrors `models/cnn_model_draft.py`):

1. Parse `(wavenumber, %T)` pairs from each `.txt` (skip lines starting with `##`).
2. Sort ascending by wavenumber.
3. Linearly interpolate onto the c8 training grid (1868 points, 399.19 - 3999.64 cm-1).
4. Per-spectrum min-max normalize to `[0, 1]`. **No log conversion** -- the
   c8 training CSV is itself in %T (values ~17..99), so the model was trained
   on normalized %T directly. The variable name `absorbance` in the training
   script is misleading.
5. Reshape to `(1, 1868, 1)`.
6. Predict with each of the 4 fold models, average the softmax probabilities,
   take argmax.

Label map (matches training): `HDPE=0  LDPE=1  PP=2  PS=3  PVC=4  PET=5`.

Source data: `C:\Users\Josem\Downloads\ftir-20260521T195159Z-3-001\ftir`
(`.ispd` files excluded -- proprietary Shimadzu binary; export to JCAMP-DX
or CSV in LabSolutions IR to include them).

## Results

| File | Prediction | Confidence | 2nd guess |
|---|---|---:|---|
| bluecap1.txt | HDPE | 100.0% | PVC (0.0%) |
| bluecap2.txt | HDPE | 100.0% | LDPE (0.0%) |
| bluecap3.txt | HDPE | 100.0% | PVC (0.0%) |
| cddisk1.txt | PS | 100.0% | PP (0.0%) |
| cddisk2.txt | PS | 100.0% | PP (0.0%) |
| cddisk3.txt | PS | 100.0% | PP (0.0%) |
| cup1.txt | PP | 98.4% | PS (1.5%) |
| cup2.txt | PP | 100.0% | PS (0.0%) |
| cup3.txt | PP | 100.0% | PS (0.0%) |
| foam1.txt | PS | 100.0% | PP (0.0%) |
| foam2.txt | PS | 100.0% | PP (0.0%) |
| foam3.txt | PS | 100.0% | PP (0.0%) |
| fruitbag1.txt | HDPE | 99.4% | LDPE (0.6%) |
| fruitbag2.txt | HDPE | 96.7% | LDPE (3.3%) |
| fruitbag3.txt | HDPE | 99.5% | LDPE (0.5%) |
| greencap.txt | HDPE | 100.0% | LDPE (0.0%) |
| greencap2.txt | HDPE | 100.0% | PVC (0.0%) |
| greencap3.txt | HDPE | 99.9% | LDPE (0.0%) |
| gum1.txt | HDPE | 100.0% | PVC (0.0%) |
| gum2.txt | HDPE | 99.9% | PVC (0.0%) |
| gum3.txt | HDPE | 100.0% | PVC (0.0%) |
| ibuprofen1.txt | PVC | 100.0% | PET (0.0%) |
| ibuprofen2.txt | PVC | 100.0% | PS (0.0%) |
| ibuprofen3.txt | PVC | 100.0% | PS (0.0%) |
| milk1.txt | HDPE | 96.4% | LDPE (3.5%) |
| milk2.txt | HDPE | 99.9% | LDPE (0.1%) |
| milk3.txt | HDPE | 99.7% | LDPE (0.3%) |
| petbluebottle1.txt | PET | 100.0% | PS (0.0%) |
| petbluebottle2.txt | PET | 100.0% | PS (0.0%) |
| petbluebottle3.txt | PET | 100.0% | PS (0.0%) |
| pvcgrey.txt | PVC | 100.0% | HDPE (0.0%) |
| pvcgrey2.txt | PVC | 99.8% | PP (0.2%) |
| pvcgrey3.txt | PVC | 99.8% | PP (0.2%) |
| pvcwhite.txt | PVC | 72.3% | HDPE (27.7%) |
| pvcwhite2.txt | PVC | 100.0% | HDPE (0.0%) |
| pvcwhite3.txt | PVC | 98.2% | HDPE (1.5%) |
| ziploc1.txt | LDPE | 100.0% | HDPE (0.0%) |
| ziploc2.txt | LDPE | 100.0% | HDPE (0.0%) |
| ziploc3.txt | LDPE | 100.0% | HDPE (0.0%) |

## Per-sample summary (replicates collapsed)

| Sample | Predicted polymer | Replicate agreement |
|---|---|---|
| bluecap | HDPE | 3/3 |
| cddisk | PS | 3/3 |
| cup | PP | 3/3 |
| foam | PS | 3/3 |
| fruitbag | HDPE | 3/3 (LDPE consistently 2nd) |
| greencap | HDPE | 3/3 |
| gum | HDPE | 3/3 |
| ibuprofen | PVC | 3/3 |
| milk | HDPE | 3/3 |
| petbluebottle | PET | 3/3 |
| pvcgrey | PVC | 3/3 |
| pvcwhite | PVC | 3/3 |
| ziploc | LDPE | 3/3 |

All 13 distinct samples produced **identical predictions across all 3 replicates**
(no replicate disagreement out of 39 spectra). 33 of 39 predictions are above
99% confidence; the lowest-confidence prediction is `pvcwhite.txt` at 72.3%
(but still PVC, consistent with the other two replicates).

## Caveats

- The CNN was trained on a single instrument's c4/c8 data. These predictions
  cross instruments (different Shimadzu IRSpirit, different atmospheric
  correction, different sampling density). High confidence does not equal
  high reliability -- a softmax of 100% can still be wrong on
  out-of-distribution inputs.
- HDPE vs LDPE distinction is genuinely difficult by FTIR; the model picks
  HDPE for the `fruitbag` and `ziploc` cases but LDPE is always the runner-up.
  Bag/film products are most commonly LDPE in real life; the HDPE call on
  fruitbag may be wrong (though ziploc-brand zipper bags do come in both).
- The 6-class softmax has no "unknown" option. Anything not in
  {HDPE, LDPE, PP, PS, PVC, PET} will still be forced into one of those
  buckets.

## Reproducing

From the repo root:

```
pip install tensorflow numpy
python predict.py "C:\path\to\ftir\folder"
```

Note: `predict.py` is git-ignored. To regenerate this table, re-run the script
and update the table here. The script includes a compatibility shim for
loading the `.keras` files on Keras 3.10 (Python 3.9).
