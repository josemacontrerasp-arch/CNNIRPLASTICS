# Saliency -- cnn-os

_with-OpenSpecy, SNV, 395 pts (804-3168 cm^-1)_

Ensemble of 5 folds; maps averaged over up to 25 correctly-classified training spectra per class.

## Per-class diagnostic-band overlap

| Class | Grad-CAM peaks (cm⁻¹) | In-range bands | Hit |
|---|---|---|---:|
| HDPE | 1440, 1506, 1548, 2046, 2346, 2394, 2868, 2934 | 2915, 2848, 1471 | 2/3 |
| LDPE | 876, 1020, 1068, 1152, 1194, 1242, 1284, 1638 | 2915, 2848, 1465, 1377 | 0/4 |
| PP | 942, 1038, 1080, 1338, 1422, 2310, 2856, 2934 | 2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840 | 3/9 |
| PS | 876, 930, 990, 1038, 1080, 1164, 1428, 2958 | 3026, 2920, 1601, 1492, 1452, 1027 | 2/6 |
| PVC | 1428, 2322, 2364, 2412, 2862, 2940, 3126, 3168 | 2912, 1427, 1331, 1254, 960 | 2/5 |
| PET | 1140, 1182, 1302, 1362, 1674, 1734, 2358, 2406 | 1715, 1409, 1241, 1094, 1017, 871 | 1/6 |

## Reproducible artifact checks

* **CO₂ band (~2349 cm⁻¹)** strongly weighted by: HDPE, PVC.
* **High-wavenumber reliance (≥2800 cm⁻¹):** HDPE 32%, LDPE 5%, PP 20%, PS 8%, PVC 22%, PET 9%.