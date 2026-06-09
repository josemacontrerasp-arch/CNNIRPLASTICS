# CNN Saliency / Explainability Analysis

Ensemble (4 fold models) gradient saliency and 1-D Grad-CAM on the 1-D CNN. For each class we average maps over 25 correctly-classified c8 spectra.

Spectra are normalized %T (absorption points downward). Diagnostic IR bands are standard polymer assignments; we check whether the model attends near them.

## Per-class diagnostic-band overlap

| Class | Grad-CAM peak centers (cm⁻¹) | Known bands | Hit |
|---|---|---|---:|
| HDPE | 2332, 2372, 2843, 3664, 3705, 3766, 3855, 3919 | 2915, 2848, 1471, 730, 719 | 1/5 |
| LDPE | 3483, 3527, 3568, 3674, 3801, 3843, 3926, 4000 | 2915, 2848, 1465, 1377, 730, 719 | 0/6 |
| PP | 1371, 1452, 2324, 2364, 2837, 2877, 2924, 2964 | 2950, 2917, 2838, 1455, 1377, 1167, 998, 973, 840 | 5/9 |
| PS | 2027, 3055, 3095, 3168, 3246, 3307, 3348, 3388 | 3026, 2920, 1601, 1492, 1452, 1027, 753, 696 | 1/8 |
| PVC | 445, 586, 698, 764, 1101, 1157, 1448, 2364 | 2912, 1427, 1331, 1254, 960, 690, 615 | 3/7 |
| PET | 403, 444, 488, 710, 1084, 1234, 1707, 2360 | 1715, 1409, 1241, 1094, 1017, 871, 722 | 4/7 |

## Reproducible artifact checks

* **Atmospheric CO₂ band (~2349 cm⁻¹):** strongly weighted (normalized Grad-CAM ≥ 0.5) for: HDPE, PP, PS, PVC. CO₂ is an environmental/instrument artifact, not a polymer feature -- any reliance here is a transfer-risk red flag.
* **High-wavenumber reliance (2800–4000 cm⁻¹, mostly shared C–H stretch + baseline):** fraction of Grad-CAM mass in that band: HDPE 42%, LDPE 69%, PP 34%, PS 33%, PVC 6%, PET 3%.
  The polyethylenes (HDPE/LDPE) lean hardest on this weakly-discriminative region, which is consistent with PE's sparse, mostly-shared band set and with the HDPE/LDPE confusions seen in the external test.


## Reading the figures

* **Background heatmap (inferno)** = Grad-CAM: bright = regions the last convolutional layer relies on for that class.
* **Cyan line** = gradient saliency: finer per-wavenumber sensitivity.
* **Dotted verticals** = textbook diagnostic bands (green = model attends near it, red = it does not).

## Takeaways

* High band-overlap means the CNN learned chemically meaningful features rather than dataset artifacts -- the regions it weights line up with the vibrational modes a spectroscopist would use.
* Where the model relies on regions *away* from diagnostic bands, treat it as a caution: it may be keying on instrument/baseline features that will not transfer across instruments (cf. the external-dataset test).

Figures: `saliency_overview.png`, `saliency_<CLASS>.png` in `results/saliency/`.