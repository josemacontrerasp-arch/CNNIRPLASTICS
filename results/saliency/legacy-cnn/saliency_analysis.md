# Saliency -- legacy-cnn

_legacy output/fold_*.keras -- per-spectrum min-max, 1868 pts (original pre-OpenSpecy pipeline)_

Ensemble of 4 folds; maps averaged over up to 25 correctly-classified training spectra per class.

## Per-class diagnostic-band overlap

| Class | Grad-CAM peaks (cm⁻¹) | In-range bands | Hit |
|---|---|---|---:|
| HDPE | 407, 449, 507, 663, 725, 1471, 2324, 2364 | 2915, 2848, 1471, 730, 719 | 3/5 |
| LDPE | 3629, 3681, 3759, 3807, 3847, 3888, 3957, 4000 | 2915, 2848, 1465, 1377, 730, 719 | 0/6 |
| PP | 399, 465, 978, 1018, 1065, 1105, 1971, 2015 | 2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840 | 2/9 |
| PS | 1367, 1811, 1851, 1928, 2004, 2090, 2131, 2920 | 3026, 2920, 1601, 1492, 1452, 1027, 753, 696 | 1/8 |
| PVC | 407, 660, 1452, 1537, 1697, 1743, 2324, 2364 | 2912, 1427, 1331, 1254, 960, 690, 615 | 1/7 |
| PET | 418, 488, 710, 862, 1005, 1084, 1230, 1707 | 1715, 1409, 1241, 1094, 1017, 871, 722 | 6/7 |

## Reproducible artifact checks

* **CO₂ band (~2349 cm⁻¹)** strongly weighted by: HDPE, LDPE, PS, PVC.
* **High-wavenumber reliance (≥2800 cm⁻¹):** HDPE 1%, LDPE 38%, PP 1%, PS 28%, PVC 25%, PET 9%.