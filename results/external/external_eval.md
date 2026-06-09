# External-Dataset Generalization Test

Soft-vote ensemble of `output/fold_1.keras` ... `fold_4.keras` (average of the 4 softmax outputs, then argmax).

Label map: `HDPE=0 LDPE=1 PP=2 PS=3 PVC=4 PET=5`. The model has **no "unknown" class** -- every spectrum is forced into one of these six.

## Headline

* **In-distribution accuracy (FLOPP-e):** 27/28 = **96.4%** (materials PE, PP, PS, PVC, PET; a PE spectrum counts correct if predicted HDPE or LDPE).

* **Out-of-distribution spectra:** 54 (bioplastics + polymers outside the six classes). These cannot be "correct"; we report the forced guess and its confidence to show whether the model over-confidently mislabels unknown polymers.

* **27/54 OOD spectra were misclassified with >=90% confidence** -- the softmax does not flag novelty, which motivates adding an explicit reject option / the new bioplastic classes.


## Per-material summary

| Material | n | In-dist? | Most common pred | Mean conf | Acc |
|---|---:|:--:|---|---:|---:|
| ABS | 3 | OOD | PS | 100% | - |
| Bamboo | 1 | OOD | PET | 92% | - |
| CPLA | 2 | OOD | PVC | 77% | - |
| Cornstarch | 1 | OOD | PP | 53% | - |
| EVA | 3 | OOD | PVC | 80% | - |
| EVOH | 5 | OOD | HDPE | 41% | - |
| Flaxstic | 1 | OOD | PS | 100% | - |
| Mater-Bi | 2 | OOD | PET | 100% | - |
| Nature Flex NK | 1 | OOD | PET | 79% | - |
| Nylon | 2 | OOD | PP | 75% | - |
| PAN | 1 | OOD | PS | 52% | - |
| PBT | 1 | OOD | PET | 100% | - |
| PC | 1 | OOD | PVC | 89% | - |
| PE | 11 | yes | HDPE | 96% | 11/11 |
| PET | 3 | yes | PET | 100% | 3/3 |
| PHA | 2 | OOD | PS | 75% | - |
| PHB | 1 | OOD | PS | 100% | - |
| PK | 1 | OOD | PS | 56% | - |
| PLA | 11 | OOD | PS | 66% | - |
| PMMA | 6 | OOD | PS | 100% | - |
| PP | 7 | yes | PP | 91% | 7/7 |
| PS | 5 | yes | PS | 100% | 4/5 |
| PU | 1 | OOD | PP | 95% | - |
| PVAc | 1 | OOD | PS | 75% | - |
| PVC | 2 | yes | PVC | 100% | 2/2 |
| PVOH | 3 | OOD | PET | 98% | - |
| SAN | 1 | OOD | PS | 100% | - |
| Sugarcane | 3 | OOD | PS | 89% | - |

## All predictions

| Dataset | File | Material | True | Pred | Conf | 2nd guess |
|---|---|---|---|---|---:|---|
| BLoP | Bamboo Bioplastic 1. Brown Straw Fragment.CSV | Bamboo | (OOD) | PET | 91.8% | PVC (8.1%) |
| BLoP | CPLA Bioplastic 1. White Coffee Cup Lid Fragment.CSV | CPLA | (OOD) | PVC | 54.4% | PS (27.5%) |
| BLoP | CPLA Bioplastic 2. White Cutlery Fragment.CSV | CPLA | (OOD) | PVC | 98.6% | PET (1.4%) |
| BLoP | Cornstarch Bioplastic 1. White Cutlery Fragment.CSV | Cornstarch | (OOD) | PP | 53.1% | PVC (39.5%) |
| BLoP | Flaxstic Bioplastic 1. Green Phone Case Fragment.CSV | Flaxstic | (OOD) | PS | 100.0% | PET (0.0%) |
| BLoP | Mater-Bi Bioplastic 1. Green Shopping Bag Film.CSV | Mater-Bi | (OOD) | PET | 100.0% | PS (0.0%) |
| BLoP | Mater-Bi Bioplastic 2. Green Produce Bag Film.CSV | Mater-Bi | (OOD) | PET | 100.0% | PS (0.0%) |
| BLoP | Nature Flex NK Bioplastic 1. Clear Cellophane Bag Film.CSV | Nature Flex NK | (OOD) | PET | 79.3% | PVC (11.3%) |
| BLoP | PHA Bioplastic 1. Blue Straw Fragment.CSV | PHA | (OOD) | PS | 100.0% | PET (0.0%) |
| BLoP | PHA Bioplastic 2. White Straw Fragment.CSV | PHA | (OOD) | PET | 50.0% | PS (49.7%) |
| BLoP | PLA Bioplastic 1. White Straw Fragment.CSV | PLA | (OOD) | PS | 74.4% | PET (25.0%) |
| BLoP | PLA Bioplastic 10. Brown Straw Fragment.CSV | PLA | (OOD) | PVC | 37.2% | PET (33.9%) |
| BLoP | PLA Bioplastic 11. Clear Cold Drink Cup Fragment.CSV | PLA | (OOD) | PS | 50.2% | PVC (29.7%) |
| BLoP | PLA Bioplastic 12. White Cutlery Wrapper Fragment.CSV | PLA | (OOD) | PET | 96.4% | PVC (3.5%) |
| BLoP | PLA Bioplastic 13. Clear Cup Lid Fragment.CSV | PLA | (OOD) | PS | 74.6% | PET (24.9%) |
| BLoP | PLA Bioplastic 14. Clear Clamshell Container Fragment.CSV | PLA | (OOD) | PS | 53.6% | PET (24.6%) |
| BLoP | PLA Bioplastic 15. White Straw Fragment.CSV | PLA | (OOD) | PS | 71.8% | PET (25.6%) |
| BLoP | PLA Bioplastic 16. White Straw Fragment.CSV | PLA | (OOD) | PS | 72.7% | PET (24.7%) |
| BLoP | PLA Bioplastic 17. Brown Straw Fragment.CSV | PLA | (OOD) | PVC | 49.9% | PS (25.1%) |
| BLoP | PLA Bioplastic 18. Clear Card Wrapper Film.CSV | PLA | (OOD) | PS | 50.7% | PET (24.9%) |
| BLoP | Sugarcane Bioplastic 1. Brown Cutlery Fragment.CSV | Sugarcane | (OOD) | PS | 100.0% | PET (0.0%) |
| BLoP | Sugarcane Bioplastic 2. Brown Straw Fragment.CSV | Sugarcane | (OOD) | PS | 66.2% | PET (33.3%) |
| BLoP | Sugarcane Bioplastic 3. Brown Straw Fragment.CSV | Sugarcane | (OOD) | PS | 100.0% | PVC (0.0%) |
| FLOPP-e | ABS-1.csv | ABS | (OOD) | PS | 100.0% | PP (0.0%) |
| FLOPP-e | ABS-2.csv | ABS | (OOD) | PVC | 99.3% | HDPE (0.6%) |
| FLOPP-e | ABS-3.csv | ABS | (OOD) | PS | 100.0% | PP (0.0%) |
| FLOPP-e | EVA-1.csv | EVA | (OOD) | LDPE | 74.2% | PVC (17.9%) |
| FLOPP-e | EVA-2.csv | EVA | (OOD) | PVC | 71.8% | PS (25.0%) |
| FLOPP-e | EVA-3.csv | EVA | (OOD) | PVC | 93.4% | HDPE (6.6%) |
| FLOPP-e | EVOH-1.csv | EVOH | (OOD) | HDPE | 36.4% | PET (26.9%) |
| FLOPP-e | EVOH-2.csv | EVOH | (OOD) | HDPE | 63.0% | PVC (20.3%) |
| FLOPP-e | EVOH-3.csv | EVOH | (OOD) | HDPE | 34.9% | PET (27.0%) |
| FLOPP-e | EVOH-4.csv | EVOH | (OOD) | HDPE | 34.6% | PET (26.9%) |
| FLOPP-e | EVOH-5.csv | EVOH | (OOD) | HDPE | 37.1% | PVC (26.1%) |
| FLOPP-e | Nylon-6.csv | Nylon | (OOD) | PP | 75.2% | HDPE (24.8%) |
| FLOPP-e | Nylon-66.csv | Nylon | (OOD) | PP | 75.1% | HDPE (24.9%) |
| FLOPP-e | PAN-1.csv | PAN | (OOD) | PS | 51.9% | PP (47.4%) |
| FLOPP-e | PBT-1.csv | PBT | (OOD) | PET | 100.0% | PS (0.0%) |
| FLOPP-e | PC-1.csv | PC | (OOD) | PVC | 89.4% | PET (10.6%) |
| FLOPP-e | PE-1.csv | PE | PE | HDPE ✓ | 88.1% | LDPE (11.8%) |
| FLOPP-e | PE-10.csv | PE | PE | HDPE ✓ | 98.2% | LDPE (1.8%) |
| FLOPP-e | PE-11.csv | PE | PE | HDPE ✓ | 99.9% | LDPE (0.0%) |
| FLOPP-e | PE-2.csv | PE | PE | HDPE ✓ | 100.0% | PVC (0.0%) |
| FLOPP-e | PE-3.csv | PE | PE | LDPE ✓ | 99.9% | HDPE (0.1%) |
| FLOPP-e | PE-4.csv | PE | PE | LDPE ✓ | 100.0% | HDPE (0.0%) |
| FLOPP-e | PE-5.csv | PE | PE | LDPE ✓ | 100.0% | HDPE (0.0%) |
| FLOPP-e | PE-6.csv | PE | PE | HDPE ✓ | 77.4% | LDPE (22.6%) |
| FLOPP-e | PE-7.csv | PE | PE | HDPE ✓ | 93.3% | LDPE (6.7%) |
| FLOPP-e | PE-8.csv | PE | PE | LDPE ✓ | 100.0% | HDPE (0.0%) |
| FLOPP-e | PE-9.csv | PE | PE | LDPE ✓ | 100.0% | HDPE (0.0%) |
| FLOPP-e | PET-1.csv | PET | PET | PET ✓ | 100.0% | PS (0.0%) |
| FLOPP-e | PET-2.csv | PET | PET | PET ✓ | 100.0% | PS (0.0%) |
| FLOPP-e | PET-3.csv | PET | PET | PET ✓ | 100.0% | PS (0.0%) |
| FLOPP-e | PHB-1.csv | PHB | (OOD) | PS | 100.0% | PET (0.0%) |
| FLOPP-e | PK-1.csv | PK | (OOD) | PS | 56.1% | PET (41.7%) |
| FLOPP-e | PLA-1.csv | PLA | (OOD) | PS | 99.5% | PET (0.3%) |
| FLOPP-e | PMMA-1.csv | PMMA | (OOD) | PS | 100.0% | PVC (0.0%) |
| FLOPP-e | PMMA-2.csv | PMMA | (OOD) | PS | 100.0% | PET (0.0%) |
| FLOPP-e | PMMA-3.csv | PMMA | (OOD) | PS | 100.0% | PET (0.0%) |
| FLOPP-e | PMMA-4.csv | PMMA | (OOD) | PS | 100.0% | PET (0.0%) |
| FLOPP-e | PMMA-5.csv | PMMA | (OOD) | PS | 100.0% | PVC (0.0%) |
| FLOPP-e | PMMA-6.csv | PMMA | (OOD) | PS | 100.0% | PVC (0.0%) |
| FLOPP-e | PP-1.csv | PP | PP | PP ✓ | 82.9% | PS (17.1%) |
| FLOPP-e | PP-2.csv | PP | PP | PP ✓ | 99.8% | PS (0.2%) |
| FLOPP-e | PP-3.csv | PP | PP | PP ✓ | 99.2% | PS (0.8%) |
| FLOPP-e | PP-4.csv | PP | PP | PP ✓ | 70.4% | PS (29.3%) |
| FLOPP-e | PP-5.csv | PP | PP | PP ✓ | 88.0% | PS (10.6%) |
| FLOPP-e | PP-6.csv | PP | PP | PP ✓ | 98.2% | PS (1.5%) |
| FLOPP-e | PP-7.csv | PP | PP | PP ✓ | 100.0% | PS (0.0%) |
| FLOPP-e | PS-1.csv | PS | PS | PS ✓ | 100.0% | PP (0.0%) |
| FLOPP-e | PS-2.csv | PS | PS | PVC ✗ | 99.2% | HDPE (0.7%) |
| FLOPP-e | PS-3.csv | PS | PS | PS ✓ | 100.0% | PP (0.0%) |
| FLOPP-e | PS-4.csv | PS | PS | PS ✓ | 100.0% | PP (0.0%) |
| FLOPP-e | PS-5.csv | PS | PS | PS ✓ | 100.0% | PP (0.0%) |
| FLOPP-e | PU-1.csv | PU | (OOD) | PP | 95.1% | PET (4.8%) |
| FLOPP-e | PVAc-1.csv | PVAc | (OOD) | PS | 74.6% | PET (25.1%) |
| FLOPP-e | PVC-1.csv | PVC | PVC | PVC ✓ | 100.0% | PET (0.0%) |
| FLOPP-e | PVC-2.csv | PVC | PVC | PVC ✓ | 100.0% | HDPE (0.0%) |
| FLOPP-e | PVOH-1.csv | PVOH | (OOD) | PET | 100.0% | PVC (0.0%) |
| FLOPP-e | PVOH-2.csv | PVOH | (OOD) | PET | 93.7% | PVC (3.9%) |
| FLOPP-e | PVOH-3.csv | PVOH | (OOD) | PET | 100.0% | PVC (0.0%) |
| FLOPP-e | SAN-1.csv | SAN | (OOD) | PS | 100.0% | PP (0.0%) |