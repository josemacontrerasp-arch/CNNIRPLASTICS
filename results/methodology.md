# Methodology

This section specifies *what* the system does and *how* it is evaluated, before
any later section explains *how it is built*. The classifier is treated here as
a **black box**: we define its input and output contract, the data it is trained
and tested on, and the exact procedure used to analyse its results. Internal
design (preprocessing, CNN/RF architecture) is deferred to the respective
implementation sections.

> Note: the detailed functional / non-functional requirements are owned by the
> Requirements section (Henry). The MoSCoW table below is a compact summary so
> the methodology is self-contained; the Requirements section is authoritative.

---

## 1. Datasets

Five spectral sources are used, in two roles: **training/validation** (data the
models may learn from and are cross-validated on) and **external test** (data
from instruments and materials never seen during training, used only to measure
generalization).

| Dataset | Role | Spectra | Classes | Grid / resolution | Provenance |
|---|---|---:|---|---|---|
| FTIR `c4` | train/val | 3 000 | 6 commodity | ~0.96 cm⁻¹, 399–4001 cm⁻¹ | Balanced reference FTIR library |
| FTIR `c8` | train/val | 3 000 | 6 commodity | ~1.93 cm⁻¹, 399–4002 cm⁻¹ | Balanced reference FTIR library |
| OpenSpecy | train/val | 8 390 FTIR | 6 + bioplastics | ~6.0 cm⁻¹, effective 804–3168 cm⁻¹ | OpenSpecy open spectral library |
| IRSpirit lab | external test | ~24 | 6 commodity | Shimadzu IRSpirit, ~630 pts | Our own lab measurements (real objects) |
| FLOPP-e | external test | 59 | 21 polymer types | 1 cm⁻¹, ~650–4000 cm⁻¹, %T | Public reference library (figshare 24593022) |
| BLoP | external test | 23 | bioplastics | ~1.93 cm⁻¹, %T | Public bioplastics library |

**Six commodity classes** are the core classification target:
PET, HDPE, LDPE, PP, PS, PVC. Work is underway to extend to **bioplastic
classes** (PLA, PHA, and the scaffolded PBAT/PBS), motivated by the external
tests below.

Key properties that shape the methodology:

* All spectra are infrared (Raman entries in OpenSpecy are dropped). Intensities
  are stored/compared as **%transmittance**; absorption appears as downward dips.
* The sources do **not** share a wavenumber axis. Before modelling, every
  spectrum is resampled onto a single shared grid (range = intersection of all
  sources, resolution = the coarsest source within that range), and any spectrum
  not fully covering the grid is dropped rather than extrapolated.
* **OpenSpecy is the resolution/range bottleneck.** Including it forces the
  shared grid from **1868 points (399–4000 cm⁻¹)** down to **395 points
  (804–3168 cm⁻¹)** — a ~79% reduction in spectral points and the loss of the
  entire 399–804 cm⁻¹ low-wavenumber fingerprint region. The trade-off between
  OpenSpecy's extra data and this domain shrinkage is evaluated explicitly
  (Section 4.3).

---

## 2. System specification (black box)

### 2.1 Input contract

* A single infrared spectrum: a set of (wavenumber [cm⁻¹], intensity) pairs.
* Arbitrary source resolution and range; the system internally resamples onto the
  trained grid and normalizes per spectrum (min–max to [0, 1]).
* Acceptable input range must overlap the trained grid; spectra far outside it
  (e.g. truncated below the trained minimum wavenumber) are out of contract.

### 2.2 Output contract

* A predicted **class label** from the supported set (six commodity classes,
  extensible to bioplastics).
* A **confidence vector** (softmax / averaged tree probabilities) over all
  classes, enabling a confidence threshold or "second-guess" reporting.
* **Known limitation:** the current models have **no explicit "unknown" class**.
  Any spectrum is forced into one of the trained classes. Novelty handling
  (a reject option / confidence threshold) is treated as a requirement and
  evaluated as a failure mode (Section 4.2).

### 2.3 Non-goals (scope boundary)

Quantification (concentration), mixture decomposition, and morphological/particle
analysis are out of scope. The system answers *"which polymer is this spectrum?"*,
not *"how much"* or *"what shape"*.

---

## 3. Requirements (MoSCoW summary)

Concise summary; see the Requirements section for the full treatment.

| Priority | Functional | Non-functional |
|---|---|---|
| **Must** | Accept an IR spectrum and output a class + confidence; classify the six commodity plastics; persist and reload trained models | High accuracy on the supported classes; reproducible evaluation |
| **Should** | Report a confidence/second-guess; evaluate on held-out and external data | Fast inference suitable for near-real-time use; robustness to noise/baseline shift |
| **Could** | Extend to bioplastic classes; flag out-of-distribution inputs (reject option) | Edge/embedded deployability; scalability to new classes/datasets |
| **Won't (this iteration)** | Quantification, mixtures, particle morphology | Certified industrial throughput guarantees |

---

## 4. Evaluation methodology

The same model-agnostic scoring is applied to every classifier (CNN, Random
Forest, cosine baseline), so results are directly comparable. Predictions are
always integer class labels plus a probability vector.

### 4.1 Metrics

* **Accuracy** — overall fraction correct.
* **Macro-F1** — unweighted mean per-class F1, so minority classes (e.g. PVC)
  are not masked by majority classes.
* **Per-class F1** and the **confusion matrix** — to locate *which* polymers are
  confused (e.g. HDPE↔LDPE), not just the headline number.
* **Confidence distribution** — used to assess over-confidence, especially on
  out-of-distribution inputs.

### 4.2 Validation protocols

1. **Cross-validation (in-domain).** 5-fold stratified CV on the training
   sources. Each spectrum is predicted exactly once by a fold model that never
   saw it. Preprocessing is per-spectrum and leakage-free, so it is applied
   before the split. *This measures in-domain fit, not transfer.*

2. **External generalization (primary signal).** Models trained only on the
   training sources are run on the **lab, FLOPP-e and BLoP** spectra, each
   resampled onto the trained grid. This is the real test of whether the model
   learned the polymer or the instrument.
   * **In-distribution external accuracy** is scored on materials in the six
     classes (FLOPP-e "PE" counts correct as HDPE *or* LDPE, since the source
     does not split them).
   * **Out-of-distribution behaviour** is reported for materials outside the six
     classes (most of BLoP; ABS, PMMA, Nylon, … in FLOPP-e). These cannot be
     "correct"; we report the forced guess and its confidence to expose whether
     the model silently mislabels unknown polymers with high confidence.

3. **Robustness.** Each test spectrum is also evaluated under controlled
   perturbations (added noise, baseline shift, corruption) to check prediction
   stability.

### 4.3 Targeted experiments

* **OpenSpecy A/B.** Train matched models with vs without OpenSpecy and compare
  *external* accuracy, to test whether OpenSpecy's coarse grid and narrowed
  domain cost more than its extra data gives. (Reported in
  `results/ab_openspecy/`.)
* **Explainability / saliency.** Gradient saliency and Grad-CAM heatmaps
  identify which wavenumber regions drive each class decision, and these are
  checked against known diagnostic IR bands. Reliance on non-diagnostic regions
  (e.g. the atmospheric CO₂ band near 2349 cm⁻¹, or shared C–H stretch regions)
  is flagged as a transfer-risk indicator. (Reported in `results/saliency/`.)

### 4.4 Reporting

For every model and protocol we report accuracy, macro-F1, the confusion matrix,
and — for external tests — the per-material prediction table and OOD confidence
summary. Baselines (cosine similarity, Random Forest) are reported alongside the
CNN so the added value of the deep model is explicit.

---

## 5. Threats to validity (brief)

* **Small external in-distribution set.** Only ~28 FLOPP-e spectra fall in the
  six classes, so single-sample differences are within noise; trends are read
  rather than point estimates.
* **%T vs absorbance.** External sources must be in the same representation as
  training (%T); an inverted (absorbance) spectrum would silently degrade
  predictions. This is checked per source before evaluation.
* **Class imbalance** (e.g. PVC scarcity in OpenSpecy) is why macro-F1 and the
  confusion matrix accompany raw accuracy throughout.
