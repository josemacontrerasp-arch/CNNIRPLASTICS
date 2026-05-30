# Cosine library-match baseline -- evaluation

Nearest-reference (centroid) classifier under cosine similarity.
This is the no-training floor any ML model must beat.

## Summary: macro-F1

| variant | within_c4_macroF1 | within_c8_macroF1 | within_OpenSpecy_macroF1 | c4->c8_macroF1 | c8->c4_macroF1 | c4->OpenSpecy_macroF1 | c8->OpenSpecy_macroF1 |
|---|---|---|---|---|---|---|---|
| raw | 0.767 | 0.792 | 0.224 | 0.575 | 0.556 | 0.024 | 0.033 |
| snv | 0.802 | 0.823 | 0.560 | 0.595 | 0.617 | 0.018 | 0.032 |
| asls | 0.781 | 0.858 | 0.453 | 0.697 | 0.695 | 0.072 | 0.072 |
| asls+snv | 0.798 | 0.810 | 0.591 | 0.714 | 0.673 | 0.069 | 0.068 |

## Within-dataset detail

### raw
- **c4**: acc=0.764, macroF1=0.767 (fold std 0.017); per-class F1: HDPE=0.84, LDPE=0.87, PET=0.88, PP=0.82, PS=0.56, PVC=0.63
- **c8**: acc=0.792, macroF1=0.792 (fold std 0.014); per-class F1: HDPE=0.76, LDPE=0.77, PET=0.87, PP=0.82, PS=0.74, PVC=0.79
- **OpenSpecy**: acc=0.312, macroF1=0.224 (fold std 0.028); per-class F1: HDPE=0.00, LDPE=0.24, PET=0.27, PP=0.62, PS=0.01, PVC=0.19

### snv
- **c4**: acc=0.806, macroF1=0.802 (fold std 0.018); per-class F1: HDPE=0.79, LDPE=0.75, PET=0.96, PP=0.84, PS=0.69, PVC=0.77
- **c8**: acc=0.822, macroF1=0.823 (fold std 0.009); per-class F1: HDPE=0.78, LDPE=0.77, PET=0.98, PP=0.80, PS=0.70, PVC=0.90
- **OpenSpecy**: acc=0.663, macroF1=0.560 (fold std 0.014); per-class F1: HDPE=0.33, LDPE=0.45, PET=0.80, PP=0.83, PS=0.69, PVC=0.26

### asls
- **c4**: acc=0.789, macroF1=0.781 (fold std 0.016); per-class F1: HDPE=0.66, LDPE=0.77, PET=1.00, PP=0.83, PS=0.68, PVC=0.75
- **c8**: acc=0.857, macroF1=0.858 (fold std 0.013); per-class F1: HDPE=0.89, LDPE=0.93, PET=0.98, PP=0.85, PS=0.67, PVC=0.83
- **OpenSpecy**: acc=0.510, macroF1=0.453 (fold std 0.012); per-class F1: HDPE=0.27, LDPE=0.38, PET=0.80, PP=0.47, PS=0.51, PVC=0.30

### asls+snv
- **c4**: acc=0.805, macroF1=0.798 (fold std 0.012); per-class F1: HDPE=0.71, LDPE=0.79, PET=0.96, PP=0.85, PS=0.68, PVC=0.80
- **c8**: acc=0.823, macroF1=0.810 (fold std 0.023); per-class F1: HDPE=0.85, LDPE=0.92, PET=0.95, PP=0.85, PS=0.50, PVC=0.79
- **OpenSpecy**: acc=0.666, macroF1=0.591 (fold std 0.017); per-class F1: HDPE=0.39, LDPE=0.47, PET=0.79, PP=0.81, PS=0.79, PVC=0.30

## Cross-dataset detail

### raw
- **c4->c8**: acc=0.572, macroF1=0.575 (dropped classes: none); per-class F1: HDPE=0.29, LDPE=0.39, PET=0.86, PP=0.81, PS=0.50, PVC=0.59
- **c8->c4**: acc=0.568, macroF1=0.556 (dropped classes: none); per-class F1: HDPE=0.23, LDPE=0.33, PET=0.89, PP=0.80, PS=0.40, PVC=0.68
- **c4->OpenSpecy**: acc=0.028, macroF1=0.024 (dropped classes: none); per-class F1: HDPE=0.00, LDPE=0.01, PET=0.05, PP=0.00, PS=0.04, PVC=0.05
- **c8->OpenSpecy**: acc=0.035, macroF1=0.033 (dropped classes: none); per-class F1: HDPE=0.01, LDPE=0.00, PET=0.03, PP=0.02, PS=0.08, PVC=0.05

### snv
- **c4->c8**: acc=0.597, macroF1=0.595 (dropped classes: none); per-class F1: HDPE=0.27, LDPE=0.30, PET=0.99, PP=0.81, PS=0.55, PVC=0.65
- **c8->c4**: acc=0.617, macroF1=0.617 (dropped classes: none); per-class F1: HDPE=0.24, LDPE=0.39, PET=0.97, PP=0.77, PS=0.57, PVC=0.75
- **c4->OpenSpecy**: acc=0.015, macroF1=0.018 (dropped classes: none); per-class F1: HDPE=0.00, LDPE=0.01, PET=0.01, PP=0.00, PS=0.04, PVC=0.05
- **c8->OpenSpecy**: acc=0.035, macroF1=0.032 (dropped classes: none); per-class F1: HDPE=0.01, LDPE=0.02, PET=0.01, PP=0.08, PS=0.03, PVC=0.06

### asls
- **c4->c8**: acc=0.722, macroF1=0.697 (dropped classes: none); per-class F1: HDPE=0.39, LDPE=0.73, PET=0.98, PP=0.86, PS=0.56, PVC=0.67
- **c8->c4**: acc=0.717, macroF1=0.695 (dropped classes: none); per-class F1: HDPE=0.73, LDPE=0.74, PET=0.97, PP=0.80, PS=0.32, PVC=0.61
- **c4->OpenSpecy**: acc=0.093, macroF1=0.072 (dropped classes: none); per-class F1: HDPE=0.12, LDPE=0.02, PET=0.02, PP=0.03, PS=0.19, PVC=0.05
- **c8->OpenSpecy**: acc=0.081, macroF1=0.072 (dropped classes: none); per-class F1: HDPE=0.15, LDPE=0.02, PET=0.05, PP=0.00, PS=0.06, PVC=0.15

### asls+snv
- **c4->c8**: acc=0.736, macroF1=0.714 (dropped classes: none); per-class F1: HDPE=0.45, LDPE=0.75, PET=0.98, PP=0.86, PS=0.57, PVC=0.68
- **c8->c4**: acc=0.721, macroF1=0.673 (dropped classes: none); per-class F1: HDPE=0.82, LDPE=0.86, PET=0.95, PP=0.80, PS=0.05, PVC=0.57
- **c4->OpenSpecy**: acc=0.089, macroF1=0.069 (dropped classes: none); per-class F1: HDPE=0.13, LDPE=0.03, PET=0.09, PP=0.03, PS=0.13, PVC=0.00
- **c8->OpenSpecy**: acc=0.078, macroF1=0.068 (dropped classes: none); per-class F1: HDPE=0.04, LDPE=0.14, PET=0.10, PP=0.00, PS=0.01, PVC=0.12
