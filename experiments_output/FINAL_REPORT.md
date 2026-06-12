# FTIR Polymer Classification — 1D-CNN vs Random Forest Study

**Scope:** six commodity plastics (HDPE, LDPE, PP, PS, PVC, PET). Models: 1-D CNN and
Random Forest (the 2-D CNN is deliberately out of scope until the configuration below is locked).
All numbers in this report come from runs in `experiments_output/` (Stages 1–4); nothing is illustrative.

---

## 1. Executive summary

The headline recommendation is concrete and a little against the grain of "more data is better":

1. **Train the shipping model on c4 + c8 only.** OpenSpecy gives no measurable transfer benefit and actively *hurts* the cleanest generalization test (lab spectra), while collapsing the spectral grid by 79%. Keep OpenSpecy (and its bioplastics) as test-only material.
2. **Use per-spectrum SNV as the preprocessing.** The configuration sweep was saturated — every reasonable option scored at ceiling in-domain — so the choice was made on transfer and robustness, where plain SNV matched or beat everything, and the first-derivative variant actively hurt.
3. **For accuracy the CNN and RF tie; for deployment the Random Forest wins.** Both reach ~99.9% cross-validated macro-F1. But the RF generalizes better to held-out lab spectra (97.4% vs 92.3%) and is dramatically safer on unknown materials.
4. **The "I don't know this material" capability should be built on the RF's confidence, not the CNN's.** At a 0.89 confidence threshold the RF keeps 97.6% of real plastics (100% of those correct) while rejecting 96.5% of out-of-distribution spectra (AUROC 0.997). The CNN's softmax is uncalibrated and useless for this (AUROC 0.836; 34 of 57 unknowns forced past 90% confidence).

If only one 2-D CNN is to be trained after this study, train it on **c4+c8, SNV, six classes**, and pair it with an RF-based confidence gate for the reject option.

---

## 2. Data, method, and the OpenSpecy grid-collapse

Five spectral sources were used. **FTIR-c4** and **FTIR-c8** are clean, balanced, lab-grade libraries (500 spectra/class each, 6 classes). **OpenSpecy** is larger (8,656 FTIR spectra after dropping 286 Raman) but coarsely and inconsistently sampled, heavily class-imbalanced, and — in the regenerated version — now also carries bioplastic labels (PLA 515, PHA 9, PBS 8, PBAT 1). **FLOPP-e** (59 spectra) is the weathered-plastics FTIR library. **BLoP** (23 spectra) is all bioplastics. The lab set (39 real-world FTIR samples) and the externals are strictly test-only.

Spectra are aligned onto a single shared wavenumber grid, then preprocessed per-spectrum (leakage-free), then evaluated with 5-fold stratified cross-validation so every training spectrum is predicted exactly once by a fold that never saw it. Held-out lab/FLOPP-e/BLoP spectra are scored by a 5-model soft-vote ensemble.

The single most important structural fact is what happens to the grid when OpenSpecy is included:

| Grid | Source(s) | Points | Range (cm⁻¹) | Step |
|---|---|---:|---|---:|
| `no_os` | c4 + c8 | 1868 | 399–4000 | 1.93 |
| `with_os` | c4 + c8 + OpenSpecy | 395 | 804–3168 | 6.00 |

Because the shared grid must be the *coarsest* common denominator, adding OpenSpecy throws away **79% of the spectral points** and deletes the entire **399–804 cm⁻¹** region. That region is not filler — it holds the PVC C–Cl stretch (~600–700), the PS aromatic out-of-plane bend (~700), and the HDPE/LDPE CH₂ rocking doublet (~720/730). This mechanism, not any single accuracy number, is the core reason to be cautious about OpenSpecy. *(See `figures/stage1_class_distribution.png`.)*

---

## 3. Does including OpenSpecy help or hurt? — **It hurts.**

This was tested with the fast RF as a proxy (Stage 2) and then confirmed with the CNN (Stage 3). The result is consistent and, importantly, the two transfer tests *disagree by design* — which is itself the finding.

| Arm | Model | CV macro-F1 | Lab acc (clean transfer) | FLOPP-e acc (confounded) |
|---|---|---:|---:|---:|
| `no_os` (c4+c8) | RF | **0.999** | **97.4% (38/39)** | 96.4% (27/28) |
| `no_os` (c4+c8) | CNN | 0.998 | 92.3% (36/39) | 96.4% (27/28) |
| `with_os` (+OpenSpecy) | RF | 0.974 | 84.6% (33/39) | 100% (28/28) |
| `with_os` (+OpenSpecy) | CNN | 0.971 | 89.7% (35/39) | 100% (28/28) |

Two things move the wrong way when OpenSpecy is added: **in-domain CV macro-F1 drops** (0.999→0.974 for RF) — the opposite of the usual "more data raises CV," and a clear sign that OpenSpecy's coarse grid and multi-instrument variability make even same-distribution classification harder — and **lab accuracy drops** (RF 97.4%→84.6%, i.e. five more of 39 lab spectra misclassified).

### The FLOPP-e extrapolation caveat (why FLOPP-e *looks* better with OpenSpecy)

The one metric that improves with OpenSpecy — FLOPP-e (96.4%→100%) — is an artifact, and it is worth being explicit about because it is genuinely misleading. FLOPP-e spectra start at ~650 cm⁻¹. On the `no_os` grid (399–4000) we are forced to **extrapolate FLOPP-e's missing 399–650 region as a flat line** — which is hot garbage, spectroscopically meaningless. We do it anyway because it is the *only* way to feed FLOPP-e to the better, OpenSpecy-free model (the model's input length is fixed at the 399-start grid). The `with_os` grid happens to start at 804 cm⁻¹, so it never touches FLOPP-e's missing region and dodges the extrapolation entirely. So `with_os` scores 100% on FLOPP-e not because OpenSpecy helped the model, but because its narrower grid sidesteps a data defect.

This is why the **lab set is the honest arbiter**: lab spectra span the full 399–3998 range, need no extrapolation, and they say OpenSpecy makes things *worse*. FLOPP-e and lab disagree, and once you understand the extrapolation confound, the disagreement resolves cleanly in favour of dropping OpenSpecy.

**Verdict:** the case against OpenSpecy is "no benefit + real, mechanistic costs" (resolution, lost fingerprint region, imbalance, 2.4× training data for nothing). Demote it to OOD/novelty test material.

---

## 4. Best preprocessing configuration — **SNV; the sweep was saturated.**

The RF sweep (Stage 2, `tables/stage2_sweep.md`, `figures/stage2_sweep.png`) over normalization (none/minmax/SNV/L2), baseline correction (AsLS/arPLS), Savitzky–Golay smoothing/derivatives, and fingerprint-region trimming returned **everything at ceiling**: CV macro-F1 0.998–1.000 and FLOPP-e 27/28 for nearly every config. The nominal "winner" (AsLS+SNV) beat the field by a single FLOPP-e spectrum and 0.001 macro-F1 — pure noise. The real finding is that **c4+c8 is so cleanly separable that preprocessing barely matters in-domain**; the only configs that dipped were L2 and fingerprint-only (26/28).

The CNN was then used to break the tie on the two configs that mattered (Stage 3):

| Config | CNN CV macro-F1 | CNN lab acc | RF lab acc |
|---|---:|---:|---:|
| SNV | 0.998 | **92.3%** | **97.4%** |
| smooth + 1st-derivative + SNV | 0.999 | 87.2% | 92.3% |

The derivative slightly raised the (already saturated) CV score but **hurt transfer for both models** — lab accuracy fell. Hypothesis: a first derivative amplifies high-frequency noise and baseline-slope differences between the training instruments (c4/c8) and the lab/FLOPP-e instruments, so it sharpens in-domain separation while making the model more sensitive to instrument shift. Plain SNV is therefore the robust choice, and it avoids the numerical fragility we saw from the baseline solvers (arPLS threw overflow warnings).

---

## 5. CNN vs RF head-to-head (c4+c8, SNV)

Both models are essentially perfect in cross-validation (`figures/stage3/no_os_norm-snv/`). The differences appear off-distribution.

| | RF | CNN |
|---|---:|---:|
| CV accuracy | 99.92% | 99.83% |
| CV macro-F1 | 0.9992 | 0.9983 |
| Lab accuracy (39 held-out) | **97.4%** | 92.3% |
| FLOPP-e in-dist (28) | 96.4% | 96.4% |
| Bioplastics forced ≥90% conf (BLoP, of 23) | **0** | 14 |

The confusion structure in-domain is benign for both. The RF's only notable leakage is a handful of PVC↔PP/PS/PET swaps (`tables/stage3_no_os__norm-snv_rf_fpfn.md`); the CNN's is a few HDPE false positives absorbing PP/PS and PP↔HDPE confusion (`..._cnn_fpfn.md`). None exceeds 0.5% of a class.

The decisive gap is **robustness**, not accuracy. The CNN matches the RF on clean plastics but is far less reliable on the lab transfer set and wildly over-confident on anything unfamiliar (next sections). For a recycling-line deployment, where the cost of a confident wrong answer on an odd fragment is high, the RF is the safer headline model. The CNN remains valuable — near-perfect on clean spectra and fast — but should be confidence-calibrated and gated.

**Strength (both):** the six commodity classes are highly separable from mid-IR bands; LDPE and PET are essentially never confused. **Weakness (CNN):** poor confidence calibration and slightly weaker instrument-shift transfer. **Weakness (RF):** marginally more PVC boundary confusion in-domain, but it recovers PVC perfectly with the full fingerprint present.

---

## 6. Weathered plastics (FLOPP-e)

On the **gradeable, in-distribution** FLOPP-e materials (PE, PP, PS, PVC, PET), the c4+c8/SNV models score **96.4% (27/28)** — a strong result given these are weathered fragments from a different instrument. The single miss and the flat-extrapolated 399–650 region (Section 3) are the only blemishes, and the latter is a data limitation, not a model failure.

The more interesting FLOPP-e signal is the **out-of-distribution** weathered polymers (ABS, EVA, EVOH, Nylon, PMMA, PC, PBT, PHB, …) that the six-class model has no label for. The CNN forces many of them into a commodity class *with high confidence* (`tables/stage4_floppe_ood_cnn.md`): PMMA→PS at 100% (6/6 ≥90%), ABS→PS, PC→PVC 100%, PBT→PET 100%, PHB→PS 100%, SAN→PS 100%. These are chemically intelligible near-misses (PMMA/PS share aromatic/ester features under SNV), but the confidence is the problem — exactly what the reject option must catch. EVOH/PVOH/PVAc, by contrast, correctly land at low confidence.

---

## 7. Bioplastic and chosen-material false positives

**BLoP bioplastics (`tables/stage4_blop_{cnn,rf}.md`).** This is the cleanest illustration of the CNN/RF safety gap. The CNN forces **14 of 23** bioplastics past 90% confidence — PLA→PVC (7/10 ≥90%, mean 91%), CPLA→PVC 99%, Mater-Bi→PET 100%, Cornstarch→PP 91%. The RF forces **0 of 23**; every bioplastic lands at low confidence (mean ~32%, max 69%), with PLA→PET at 40% (a chemically sensible weak guess via the shared ester C=O). The RF is naturally uncertain on material it has never seen; the CNN is confidently wrong.

**NIST near-miss probes (`tables/stage4_nist_probes.md`).** These are deliberate adversarial inputs — small molecules sharing key bands with a target class:

| Probe | shares bands with | CNN → | RF → |
|---|---|---|---|
| Stearic acid | HDPE/LDPE (CH₂ chain) | HDPE @ 93.3% | PVC @ 45.9% |
| Ethylbenzene | PS (aromatic) | HDPE @ 100% | PVC @ 51.0% |
| Ethyl acetate | PET (ester) | HDPE @ 100% | PVC @ 52.4% |

Again the CNN is confidently wrong (and interestingly collapses everything to HDPE at saturation), while the RF stays near its ~50% uncertainty floor and is trivially rejected by a threshold. **Note:** the fourth intended probe, **polyisobutylene (Vistanex)**, failed to load — the NIST JCAMP for that entry interpolated to a flat line (likely a gas-phase or malformed record). It is excluded from the numbers above; supplying a measured PIB/Vistanex spectrum (or a different NIST entry) would complete the PP-proxy probe. This is the one gap in the stress test and is easy to close.

---

## 8. Optimal "other / unknown" confidence threshold, and alternatives

This is the question with the most actionable answer. Treating max class-probability as a novelty score and scoring 6,000 known spectra against 57 OOD spectra (BLoP + FLOPP-e OOD + NIST):

| Model | AUROC (known vs OOD) | Recommended threshold | Known kept | Accuracy of kept | OOD rejected |
|---|---:|---:|---:|---:|---:|
| **RF** | **0.997** | **0.89** | 97.6% | 100% | 96.5% |
| CNN | 0.836 | — (none viable) | 0% | n/a | 100% only by rejecting all |

The RF result is deployment-ready: a single threshold at **0.89** rejects ~96% of unknown materials while losing only ~2% of real plastics — and every plastic it *does* accept is correctly classified. The CNN's softmax cannot do this: its known and unknown confidence distributions overlap (AUROC 0.836), and the only threshold that rejects most OOD also rejects all the knowns. *(See `figures/stage4_roc.png` and `figures/stage4_threshold_curves.png`.)*

**Other ways to get an "other" class — tested, with a clear loser.** We also trained an explicit 7th `OTHER` class (RF on the with-OpenSpecy data, using its PLA/PHA/PBS spectra as OTHER). In-distribution it learned that class reasonably (F1 0.91), **but it routed 0 of 23 external BLoP bioplastics and 0 of 3 NIST probes to OTHER.** The lesson is important: a closed trained class learns "these specific bioplastics from this instrument," not "anything that isn't my six classes," so it does not generalize to *novel* unknowns. Confidence thresholding does. Other viable directions worth a line each: temperature scaling / Platt calibration to make the CNN's softmax usable; an energy or max-logit OOD score; MC-dropout or deep-ensemble disagreement for the CNN; or a one-class/Mahalanobis novelty model on the penultimate features. Given the data in hand, **RF max-probability thresholding at ~0.89 is the simplest method that actually works.**

---

## 9. Strengths, weaknesses, and recommendations

**Where the strength comes from — the data.** c4+c8 are clean, balanced, full-range, and the six classes are genuinely separable in the mid-IR. That is why both models hit ~99.9% and why preprocessing barely moves the needle. The strength is the dataset, not the model sophistication.

**Where the weaknesses are.** (1) *OpenSpecy* — coarse, imbalanced, range-limited; its only effect here is to collapse the grid and lose the fingerprint region. (2) *FLOPP-e's truncated low-wavenumber range* — forces meaningless extrapolation for the full-range model and confounds cross-arm comparison; the lab set is the trustworthy transfer benchmark. (3) *The CNN's confidence calibration* — near-perfect on clean spectra, badly over-confident on everything else, which is a real deployment liability on a heterogeneous waste stream. (4) *Small external test sets* — 28 gradeable FLOPP-e and 39 lab spectra; trends are clear but single-spectrum swings should not be over-read.

**Recommendations / next steps:**

1. **Lock the configuration for the single 2-D CNN run:** c4+c8 only, six classes, per-spectrum SNV, 1868-point 399–4000 grid. This study exists precisely so that the 2-D CNN is trained once, on the right setup.
2. **Deploy the RF as the classifier + reject gate** at a 0.89 confidence threshold; treat anything below as "unknown — not one of the six."
3. **If the CNN is to be used in production, calibrate it first** (temperature scaling on a held-out set), then re-evaluate its AUROC for the reject option; do not ship its raw softmax as a confidence.
4. **Close the stress-test gap:** supply a real Vistanex/PIB spectrum, and consider widening the FLOPP-e comparison or adding more lab spectra to tighten the transfer estimate.
5. **Keep OpenSpecy and bioplastics as a permanent OOD test suite**, not as training data.

---

## 10. Files and reproducibility

All artifacts live under `experiments_output/`:

- `models/` — saved fold models per run. **The headline/shipping CNN is the 5-fold soft-vote ensemble in `models/no_os__norm-snv/cnn_fold_*.keras` (+ `oof.npz`)** — not the `smooth+d1+snv` or `with_os` runs. Run `python -m experiments.save_final_models` to consolidate the deployment artifacts into `models/final/`: the recommended **RF** (`rf_final.pkl` + the 5 CV forests), the headline **CNN** ensemble (`cnn_fold_*.keras`), and `deployment.json` with the class order, preprocessing, grid, and 0.89 reject threshold.
- `figures/` — `stage1_class_distribution.png`, `stage2_sweep.png`, `stage3/<run>/` (confusion matrices in-domain + lab, per-class bar charts for CNN and RF), `stage4_roc.png`, `stage4_threshold_curves.png`.
- `tables/` — preprocessing sweep, OpenSpecy A/B, per-run FP/FN tables, BLoP/FLOPP-e/NIST OOD tables, reject-option summary.
- `results/` — machine-readable JSON for every stage.

Pipeline (each stage is one command from the project root in `CNN_env`): `stage1_summary` → `stage2_rf_sweep` → `stage3_headtohead` → `stage4_stress`. Stages 1–2 and 4 are CPU-only; Stage 3 trains the CNN (GPU advised). See `experiments/README.md`.

*Note: some downloaded files carry a " (1)" suffix from the browser (e.g. `cnn_fold_1 (1).keras`); the loader's `cnn_fold_*.keras` glob still matches them, so they work as-is.*
