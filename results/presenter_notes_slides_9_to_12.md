# Presenter Notes — Slides 9 to 12 of *IR Classifier*

Talking points, deeper context, and likely audience questions. Use as a
cheat sheet, not a script.

---

## Slide 9 — Divergent → Convergent

### What's on the slide
- Lead line: "First FTIR lab done. CNN ensemble tested on 39 (13 x 3)
  IRspirit spectra. All samples correctly classified with mean
  confidence of 99 percent."
- Results table with 13 samples and per-sample mean confidence.

### What to say beyond the bullets

1. **What was actually tested.** We collected 13 everyday plastic items
   ourselves (bottle caps, milk jug, foam, CD case, water bottle, PVC
   pipe, ibuprofen blister pack, gum wrapper, ziploc, fruit bag, cup),
   measured each one three times on the Shimadzu IRSpirit, and ran every
   spectrum through our trained CNN.

2. **The "ensemble" detail.** We have 4 saved fold models from the
   stratified k-fold cross-validation. For each test spectrum we run all
   4 models and **average their softmax probability vectors** before
   taking the argmax. This soft-voting reduces variance compared to
   trusting any single model.

3. **Why "99 percent mean confidence" matters less than it sounds.** A
   neural net with softmax output can be confidently wrong. The
   important number is actually the **39 / 39 replicate agreement** — the
   same physical sample measured three times never confused the model.
   That tells you the model is **stable**, not just confident.

4. **This is a cross-instrument test.** The CNN was trained on the c4 +
   c8 in-house datasets (different instrument). It had never seen an
   IRSpirit spectrum in training. The fact that it still classified
   correctly is the real takeaway — it generalized across hardware.

5. **The preprocessing fix worth mentioning.** The training script's
   variable was called `absorbance`, but the c8 CSV values range from
   17 to 99, meaning the data was actually in **percent
   transmittance**. We caught this when our first prediction run gave
   nonsense (everything classified as PVC). Switched to normalizing %T
   directly — matched the training pipeline — and got the 99 percent
   results.

6. **Worst sample to call out (honest framing).** `pvcwhite` had only
   72 percent confidence on replicate 1 (still PVC). All other 38
   spectra were above 96 percent. Useful to mention so the audience sees
   you're not sweeping anything under the rug.

### Likely questions

- **"Could the model just be biased toward HDPE / PVC?"**
  No — predictions cover 5 of the 6 classes (HDPE, LDPE, PP, PS, PET,
  PVC); only HDPE and PVC happen to appear more often because more of
  the lab samples are made of those plastics in real life.

- **"Aren't 13 samples too few?"**
  Yes, this is preliminary. Each sample has 3 replicates so it's
  39 spectra of evidence, but the next step is more samples and harder
  cases (weathered, mixed, pigmented).

- **"Could the high confidence be overfitting?"**
  The model was trained with 5-fold cross-validation specifically to
  detect overfitting — accuracy was high across all folds, not just
  one. The IRSpirit test set is also entirely new data, never seen
  during training or validation.

---

## Slide 10 — Challenges

### What's on the slide
- Inconsistent file formats (.txt variations, delimiters, headers)
- Label inconsistency across polymer naming
- Training time started very long, optimized but still hours
- Data: diverse IR spectra with equal sample distribution per class

### What to say beyond the bullets

1. **The file-format zoo.** FTIR exports look standardized but aren't.
   JCAMP-DX, plain CSV, tab-separated, space-separated, with-header,
   without-header, low-to-high wavenumber, high-to-low. We had to write
   a parser that strips comment lines (lines starting with `##`),
   detects the delimiter, and re-sorts ascending. This kind of "data
   janitor" work isn't glamorous but it is most of the project.

2. **Label normalization is harder than it looks.** PET alone appears
   in metadata as: "PET", "PETE", "PET-G", "PETG", "Polyethylene
   Terephthalate", "poly(ethylene terephthalate)", "polyester". HDPE
   appears as "HDPE", "high density polyethylene", "high-density
   polyethylene", "polyethylene_high_density". We built a regex
   dictionary in the R dataset builder to map all variants into 6
   canonical classes.

3. **Why training was slow.** 1D CNN with two conv blocks, three dense
   layers, up to 1000 epochs per fold, batch size 64, 5 folds. Early
   stopping with patience 100 on validation loss. On CPU each fold can
   take 1–2 hours; on a GPU we run folds **serially** to avoid memory
   issues — comment in `cnn_model_draft.py` literally says "I will run
   each fold separately to avoid GPU memory issues."

4. **Class balance choice.** Our two in-house training datasets
   (FTIR_PLASTIC_c4 and c8) are deliberately balanced — 500 samples per
   class × 6 classes × 2 files = 6000 spectra. We did **not** train on
   OpenSpecy (8390 spectra) even though it's bigger, because it's
   imbalanced (PP has 2451, PVC has 257). Training on it would bias the
   model. We reserve OpenSpecy for stress-testing generalization.

5. **The c4 / c8 grid mismatch.** c4 was sampled at twice the
   resolution of c8 (twice as many wavenumber points). We took every
   other point of c4 to match c8's spacing. Wavenumber alignment was
   verified manually.

### Likely questions

- **"Did you discard any data?"**
  Spectra that were entirely NaN, and spectra whose wavenumber spacing
  deviated by more than 2 percent from the expected uniform grid (the
  script raises an error). Otherwise no.

- **"Why not just use OpenSpecy directly?"**
  Class imbalance, mixed instruments, mixed acquisition conditions.
  Good for testing robustness, bad as a sole training source.

- **"How long does inference take?"**
  Milliseconds per spectrum. The hours are training only.

---

## Slide 11 — Next steps

### What's on the slide
- Filtering — adding noise vs removing
- Mixing datasets to introduce variability
- Running different preprocessing alternatives (filtering + baseline
  correction) and dataset mixes
- Analyzing model performance to change CNN architecture

### What to say beyond the bullets

1. **"Adding noise vs removing" — both have a purpose.**
   - *Removing* noise: smoothing filters (Savitzky-Golay) clean the
     spectrum before training. Helps the model focus on real peaks.
   - *Adding* noise: data augmentation — perturb training spectra with
     Gaussian noise so the model becomes robust to real instrument
     noise. Counter-intuitive but standard practice.
   - We want to test which actually improves cross-instrument accuracy.

2. **Baseline correction.** FTIR spectra often have a sloped or curved
   baseline caused by scattering and sample thickness. Standard fixes:
   asymmetric least squares (ALS), rubber-band correction, polynomial
   detrending. Our current pipeline does only min-max normalization,
   which handles intensity but not shape distortion.

3. **Mixing datasets for variability.** Right now we train on c4 + c8
   only. Plan: combine c4, c8, OpenSpecy (the diverse one), and our
   own IRSpirit lab measurements. Goal — make the model robust to the
   instrument it's actually deployed on.

4. **CNN architecture changes worth exploring.**
   - Adding **batch normalization** between conv layers (we have none).
   - Wider kernels (5 or 7 instead of 3) for broader peak features.
   - 1D ResNet-style residual connections — better gradient flow,
     deeper networks.
   - **Attention** over the spectral axis to focus on diagnostic peak
     regions (carbonyl ~1715, C-H stretches ~2900, C-Cl ~700 etc).

5. **Why "analyzing model performance" matters.** Without confusion
   matrices and per-class precision/recall we don't know *where* the
   model fails. The next pass will produce these per fold so we can
   target the weak class instead of changing the model blindly.

### Likely questions

- **"Why not just collect more lab data?"**
  Time and cost. Each FTIR session is bench time on a shared instrument.
  Augmentation gets cheap performance gains while we scale up real
  measurements.

- **"Which architecture change would you try first?"**
  Batch normalization. Cheapest change, biggest impact in practice
  for 1D conv nets.

---

## Slide 12 — Remaining Questions

### What's on the slide
- Lab data → real-world performance gap
- Behavior on erroneous input
- Polymer identity vs instrument identity
- Optimal uncertainty threshold / "I don't know" option

### What to say beyond the bullets

1. **Lab vs real-world gap.** Lab samples are pristine: pure polymer,
   thick enough for good signal, no weathering. Real microplastics in
   the field have UV degradation, biofilms, fillers, pigments,
   plasticizers, mixed-polymer composites. Performance will degrade —
   the question is how much, and which class boundaries collapse
   first. Likely candidates: HDPE / LDPE (already a hard pair), and
   PET / PVC under certain pigments.

2. **Erroneous input behavior.** Today the model outputs a softmax
   over 6 classes — it has no choice but to pick one. If we feed it a
   Raman spectrum, a noise vector, or a cellulose spectrum, it will
   still confidently return HDPE / LDPE / PP / PS / PVC / PET. We need
   either (a) a 7th "unknown" class trained on out-of-distribution
   data, or (b) a separate OOD detector running in parallel.

3. **Polymer vs instrument identity — the real worry.**
   This is a classic ML failure: the model could be picking up on the
   *instrument fingerprint* (atmospheric CO2 bands, detector noise
   shape, baseline curvature characteristic of a specific machine)
   rather than the polymer's vibrational modes. Our IRSpirit cross-
   instrument result is a *positive signal* against this, but not
   conclusive. To really test it, we'd need to compare layer
   activations across instruments, or use SHAP / Grad-CAM to confirm
   the model attends to chemically meaningful peaks.

4. **The "I don't know" threshold — concrete approaches.**
   - **Maximum softmax probability (MSP):** simplest. Reject if
     `max(softmax) < t`. Easy but neural nets are overconfident.
   - **Temperature scaling:** post-hoc calibration of softmax —
     divide logits by a learned scalar. Better-calibrated confidences.
   - **Monte Carlo dropout:** run inference N times with dropout on,
     use the variance across runs as an uncertainty estimate.
   - **Mahalanobis or energy-based scoring** on hidden-layer
     features — better separates in-distribution from OOD inputs.
   - The threshold itself has to come from a **held-out calibration
     set** where you know the right answers. Typical range 80–90
     percent MSP, with a "reject zone" between, say, 70 and 90.

### Likely questions

- **"How do you know it's not learning the instrument?"**
  Honest answer — we don't, fully. The IRSpirit cross-instrument
  result is encouraging but not proof. Future work: explainability
  methods (SHAP, Grad-CAM) to verify peak-level attention.

- **"What confidence threshold would you pick today?"**
  We don't know yet. Need a calibration dataset. Tentative starting
  point: above 90 percent = accept, below 70 percent = reject, in
  between = flag for human review.

- **"What's the worst-case failure mode?"**
  Silent misclassification — the model confidently labels a mixed or
  unknown polymer as one of the six classes, and downstream decisions
  trust that label. The "unknown" class plus threshold work directly
  addresses this.

---

## General presenter tips

- When the table on slide 9 is up, **point at the worst row first**
  (`pvcwhite 90.2 %`) — shows honesty, then frame the rest as the
  good news. Audiences trust speakers who lead with caveats.

- On slides 10–11, frame "challenges" and "next steps" as **connected
  pairs**. The next-steps slide is the response to the challenges
  slide.

- On slide 12, **don't memorize all four uncertainty methods**. Pick
  one (Monte Carlo dropout is the easiest to explain) and have it
  ready as your default answer. Mention the others only if pressed.

- If anyone asks for code: the repository is at
  `github.com/josemacontrerasp-arch/CNNIRPLASTICS` — the training
  script is `models/cnn_model_draft.py`, the prediction pipeline is
  `predict.py` (gitignored), and the results table is at
  `results/predictions.md`.
