# Saliency -- final-cnn

_deployment CNN -- no-OpenSpecy, SNV, 1868 pts_

Ensemble of 5 folds; maps averaged over up to 25 correctly-classified training spectra per class.

## Per-class diagnostic-band overlap

| Class | Grad-CAM peaks (cm⁻¹) | In-range bands | Hit |
|---|---|---|---:|
| HDPE | 407, 447, 503, 544, 660, 710, 2320, 2360 | 2915, 2848, 1471, 730, 719 | 2/5 |
| LDPE | 2121, 2173, 2224, 2393, 2983, 3759, 3863, 3996 | 2915, 2848, 1465, 1377, 730, 719 | 0/6 |
| PP | 399, 461, 974, 1086, 1126, 1969, 2011, 2920 | 2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840 | 3/9 |
| PS | 690, 752, 1373, 1452, 2399, 2440, 2480, 2925 | 3026, 2920, 1601, 1492, 1452, 1027, 753, 696 | 4/8 |
| PVC | 407, 660, 1452, 1506, 1552, 1743, 2320, 2360 | 2912, 1427, 1331, 1254, 960, 690, 615 | 1/7 |
| PET | 418, 488, 710, 858, 1005, 1080, 1230, 1703 | 1715, 1409, 1241, 1094, 1017, 871, 722 | 6/7 |

## Reproducible artifact checks

* **CO₂ band (~2349 cm⁻¹)** strongly weighted by: HDPE, PVC.
* **High-wavenumber reliance (≥2800 cm⁻¹):** HDPE 2%, LDPE 54%, PP 15%, PS 19%, PVC 13%, PET 35%.