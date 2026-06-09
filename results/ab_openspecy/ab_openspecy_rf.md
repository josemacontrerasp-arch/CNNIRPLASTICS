# A/B: OpenSpecy inclusion vs external generalization (RF)

Both arms: six commodity classes, per-spectrum min-max, identical model. Only the training data / grid that OpenSpecy forces differs.

| Metric | without OpenSpecy | with OpenSpecy |
|---|---|---|
| Training spectra | 6000 | 14124 |
| Grid points | 1868 | 395 |
| Grid range (cm⁻¹) | 399–4000 | 804–3168 |
| CV accuracy (held-out folds) | 99.85% | 97.26% |
| **External in-dist acc (FLOPP-e)** | **96.4%** (27/28) | **100.0%** (28/28) |
| OOD forced @≥90% conf | 0/54 | 0/54 |

## Interpretation

* **CV accuracy** is an in-domain score (held-out folds of the *same* datasets); it tends to stay high or even rise with more data and says little about transfer.
* **External in-distribution accuracy** is the real generalization signal -- spectra from instruments never seen in training.
* **Grid collapse:** forcing OpenSpecy in changes the shared grid from 1868 pts (399-4000 cm⁻¹) to 395 pts (804-3168 cm⁻¹) -- a 79% drop in spectral points and the loss of the entire 399-804 cm⁻¹ low-wavenumber fingerprint region. This is the concrete mechanism behind the team's concern, independent of accuracy.
* Net effect of adding OpenSpecy on external accuracy: **+3.6 pp** (96.4% -> 100.0%), i.e. 1 spectrum of 28. With only 28 gradeable external spectra (all easy commodity polymers), this difference is within noise -- the RF arm neither confirms nor refutes the hypothesis on accuracy alone.
* RF stays under the 90%-confidence bar on every OOD spectrum, whereas the CNN forced 27/54 OOD spectra past it (results/external): the forest is markedly less over-confident on unknown polymers than the CNN.
* **Headline model still pending:** run `--model cnn` on Colab/GPU to get the 1-D CNN arm. The existing saved fold_*.keras (pre-OpenSpecy) already give the WITHOUT arm (96.4% on FLOPP-e); the CNN WITH-OpenSpecy arm is the missing half and is what the report should headline.