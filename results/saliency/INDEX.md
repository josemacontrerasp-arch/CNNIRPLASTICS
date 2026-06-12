# CNN Saliency Analysis -- all families

Gradient saliency + 1-D Grad-CAM per CNN family, each using its own training preprocessing and wavenumber grid.

| Family | Configuration | Folder |
|---|---|---|
| final-cnn | deployment CNN -- no-OpenSpecy, SNV, 1868 pts | [`final-cnn/`](final-cnn/saliency_analysis.md) |
| cnn-d1 | no-OpenSpecy, smooth + 1st-derivative + SNV, 1868 pts | [`cnn-d1/`](cnn-d1/saliency_analysis.md) |
| cnn-os | with-OpenSpecy, SNV, 395 pts (804-3168 cm^-1) | [`cnn-os/`](cnn-os/saliency_analysis.md) |
| legacy-cnn | legacy output/fold_*.keras -- per-spectrum min-max, 1868 pts (original pre-OpenSpecy pipeline) | [`legacy-cnn/`](legacy-cnn/saliency_analysis.md) |

Each folder has `saliency_<CLASS>.png` (overlay), `saliency_overview.png` (six classes stacked), and `saliency_analysis.md` (band-overlap + artifact checks).