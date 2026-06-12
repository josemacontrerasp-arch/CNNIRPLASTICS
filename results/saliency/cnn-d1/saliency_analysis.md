# Saliency -- cnn-d1

_no-OpenSpecy, smooth + 1st-derivative + SNV, 1868 pts_

Ensemble of 5 folds; maps averaged over up to 25 correctly-classified training spectra per class.

## Per-class diagnostic-band overlap

| Class | Grad-CAM peaks (cm⁻¹) | In-range bands | Hit |
|---|---|---|---:|
| HDPE | 702, 1471, 1751, 2306, 2376, 2819, 2860, 2929 | 2915, 2848, 1471, 730, 719 | 5/5 |
| LDPE | 717, 1377, 1471, 2372, 2862, 2931, 2974, 3666 | 2915, 2848, 1465, 1377, 730, 719 | 6/6 |
| PP | 970, 1167, 1377, 1464, 2376, 2850, 2931, 2972 | 2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840 | 8/9 |
| PS | 698, 760, 908, 1383, 1452, 1495, 2939, 2979 | 3026, 2920, 1601, 1492, 1452, 1027, 753, 696 | 5/8 |
| PVC | 1558, 1651, 1751, 2376, 2939, 3664, 3762, 3919 | 2912, 1427, 1331, 1254, 960, 690, 615 | 1/7 |
| PET | 725, 874, 1016, 1124, 1265, 1342, 1414, 1732 | 1715, 1409, 1241, 1094, 1017, 871, 722 | 6/7 |

## Reproducible artifact checks

* **CO₂ band (~2349 cm⁻¹)** strongly weighted by: PVC.
* **High-wavenumber reliance (≥2800 cm⁻¹):** HDPE 49%, LDPE 53%, PP 38%, PS 23%, PVC 30%, PET 10%.