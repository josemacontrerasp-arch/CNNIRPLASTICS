# Real-world test — all model families

30 spectra from `C:\Users\Josem\Downloads\data`  |  reject rule: final-rf max prob < 0.89 -> UNKNOWN

| file | deployment (RF+reject) | legacy-cnn | legacy-rf | final-rf | final-cnn | cnn-d1 | cnn-os |
|---|---|---|---|---|---|---|---|
| allumium | **UNKNOWN** | PVC 100% | PP 31% | PVC 60% | PVC 47% | PVC 100% | PET 73% |
| background | **UNKNOWN** | HDPE 100% | HDPE 26% | PVC 54% | PVC 100% | PVC 100% | PET 83% |
| blaclpla | **UNKNOWN** | PVC 75% | PP 30% | PET 37% | PS 50% | HDPE 55% | PS 60% |
| blaclpla2 | **UNKNOWN** | PVC 75% | PP 31% | PET 37% | PVC 55% | HDPE 72% | PS 60% |
| blaclpla3 | **UNKNOWN** | PVC 75% | PP 30% | PET 37% | PVC 57% | HDPE 39% | PS 60% |
| brown | **UNKNOWN** | PVC 78% | PP 24% | PVC 67% | HDPE 100% | HDPE 79% | HDPE 70% |
| brown2 | **UNKNOWN** | PVC 82% | PP 24% | PVC 67% | HDPE 99% | HDPE 82% | HDPE 51% |
| greypla | **UNKNOWN** | PVC 75% | PP 30% | PET 37% | PVC 52% | HDPE 76% | PS 60% |
| greypla2 | **UNKNOWN** | PVC 75% | PP 32% | PET 37% | PVC 60% | PET 41% | PS 60% |
| greypla3 | **UNKNOWN** | PVC 75% | PP 31% | PET 37% | PVC 59% | HDPE 74% | PS 59% |
| greypla4 | **UNKNOWN** | PVC 75% | PP 30% | PET 37% | PVC 59% | HDPE 45% | PS 59% |
| ldpewet | **HDPE** | PET 41% | PS 25% | HDPE 96% | HDPE 100% | HDPE 67% | HDPE 79% |
| orangerubberband | **UNKNOWN** | PVC 100% | PP 26% | PP 77% | PP 100% | PP 100% | PP 98% |
| orangerubberband2 | **UNKNOWN** | PVC 100% | PP 33% | PP 77% | PP 100% | PP 76% | PP 99% |
| oreo | **UNKNOWN** | PVC 100% | PP 37% | PVC 64% | PVC 100% | PVC 100% | PP 43% |
| oreo2 | **UNKNOWN** | PVC 100% | PP 37% | PVC 62% | PVC 88% | PVC 100% | PVC 50% |
| paper1 | **UNKNOWN** | PVC 74% | PET 33% | PVC 73% | HDPE 78% | PVC 58% | HDPE 96% |
| pop | **UNKNOWN** | PET 50% | HDPE 28% | LDPE 76% | LDPE 100% | LDPE 100% | LDPE 98% |
| pop2 | **UNKNOWN** | PET 53% | HDPE 28% | LDPE 80% | LDPE 100% | LDPE 100% | LDPE 97% |
| pvcscrepe | **PVC** | PVC 100% | PP 37% | PVC 92% | PVC 100% | PVC 100% | PVC 100% |
| TPU | **UNKNOWN** | PVC 100% | HDPE 27% | PET 45% | PET 37% | PVC 61% | PET 96% |
| TPU2 | **UNKNOWN** | PVC 100% | HDPE 26% | PET 45% | PVC 38% | PVC 62% | PET 95% |
| TPU3 | **UNKNOWN** | PVC 100% | HDPE 27% | PET 44% | PS 35% | PVC 68% | PET 97% |
| wetcd | **PS** | PVC 54% | PP 31% | PS 89% | PS 100% | PS 100% | PS 100% |
| wetpla | **UNKNOWN** | PVC 75% | PP 31% | PET 32% | PVC 79% | HDPE 94% | PS 51% |
| wetpp | **UNKNOWN** | PVC 34% | HDPE 24% | PP 69% | PP 96% | PP 100% | PP 87% |
| whitetextile | **UNKNOWN** | HDPE 75% | PP 25% | PVC 61% | HDPE 100% | HDPE 59% | HDPE 69% |
| whitetextile2 | **UNKNOWN** | HDPE 100% | PET 22% | PVC 61% | HDPE 100% | HDPE 68% | HDPE 56% |
| yellowpap | **UNKNOWN** | HDPE 100% | PP 25% | PP 57% | PP 81% | PP 100% | PP 96% |
| yellowpapturned | **UNKNOWN** | PVC 100% | HDPE 27% | PVC 58% | PVC 98% | PVC 100% | PVC 99% |

Saliency maps: `saliency/<file>.png` (Grad-CAM background + gradient saliency, final CNN ensemble, predicted class); `saliency_overview.png` stacks all Grad-CAMs.
