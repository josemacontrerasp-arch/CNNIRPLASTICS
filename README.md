# CNNIRPLASTICS

This repository now only contains multiple FTIR spectral datasets used for polymer classification and machine learning experiments.

The datasets are primarily focused on the following six commodity plastics:

- PET — Polyethylene Terephthalate
- HDPE — High-Density Polyethylene
- LDPE — Low-Density Polyethylene
- PP — Polypropylene
- PS — Polystyrene
- PVC — Polyvinyl Chloride

---

# Dataset Overview

| Dataset | Samples | Description |
|---|---:|---|
| `openspecy_polymer_dataset.csv` | 8390 | Extracted from the OpenSpecy spectral library |
| `FTIR_PLASTIC_c4.csv` | 3000 | Balanced FTIR plastics dataset |
| `FTIR_PLASTIC_c8.csv` | 3000 | Balanced FTIR plastics dataset |

---

# 1. OpenSpecy Dataset

## File

```text
openspecy_polymer_dataset.csv
```

## Description

This dataset was generated from the OpenSpecy spectral library using custom building scripts.

The dataset contains:
- FTIR and related infrared spectra
- 1983 spectral features per sample
- 6 polymer classes

Each row corresponds to one spectrum.

---

## Dataset Shape

```text
8390 rows × 1984 columns
```

- Columns `V1` → `V1983`:
  spectral intensity values
- Column `label`:
  polymer class label

---

## Class Distribution

| Polymer | Samples |
|---|---:|
| HDPE | 1057 |
| LDPE | 867 |
| PET | 2267 |
| PP | 2451 |
| PS | 1491 |
| PVC | 257 |

---

## Spectral Features

The 1983 feature columns correspond to spectral intensity values measured across different wavenumbers.

Additional wavenumber metadata is stored separately in:

```text
openspecy_wavenumbers.csv
```

Each feature index corresponds to a specific FTIR wavenumber.

---

# 2. FTIR Plastics Dataset

## Files

```text
FTIR_PLASTIC_c4.csv
FTIR_PLASTIC_c8.csv
```

---

## Description

These datasets are balanced FTIR polymer datasets containing 1000 samples for each plastic class.

Each sample stores:
- polymer metadata
- spectral wavelength values
- spectral intensity values

---

## Dataset Shape

Each dataset contains:

```text
3000 samples
```

with balanced polymer classes.

---

## Columns

| Column | Description |
|---|---|
| `IDE` | Sample identifier |
| `Polymer` | Polymer class label |
| `Technic` | Spectroscopy technique |
| `Sample` | Sample description |
| `BR` | Metadata field |
| `RST` | Metadata field |
| `Data(x)` | Wavelength / wavenumber values |
| `Data(y)` | Spectral intensity values |

---

## Important Note About Spectral Data

In FTIR spectroscopy:
- `Data(x)` usually represents the wavenumber axis (cm⁻¹)
- `Data(y)` represents absorbance, transmittance, or intensity values

The exact interpretation depends on the original acquisition setup.

---

## 3. Functional Requirements

### 3.1 Spectral Input Processing

The system shall:

* Accept FTIR spectral data as input.
* Preprocess spectra into a standardized format suitable for machine learning.
* Normalize absorbance values before classification.

---

### 3.2 Plastic Classification

The system shall:

* Predict the plastic category from a given infrared spectrum.
* Support classification into the six defined plastic classes.
* Output the predicted class and confidence score.

---

### 3.3 Machine Learning Models

The project shall implement and evaluate:

* A 1D Convolutional Neural Network (1D-CNN)
* A Random Forest classifier

The models shall be compared using:

* Classification accuracy
* Inference speed
* Robustness
* Suitability for real-time deployment

---

### 3.4 Model Persistence

The system shall:

* Save trained models to disk.
* Reload trained models for future inference without retraining.

---

### 3.5 Evaluation

The system shall:

* Evaluate models using validation and test datasets.
* Produce accuracy metrics and classification reports.
* Support single-sample testing for inference verification.

---

# 4. Non-Functional Requirements

### 4.1 Accuracy

The classifier should achieve high classification accuracy across all supported plastic categories.

(Done) Current experimental results using the 1D-CNN model have achieved near-perfect classification performance on the available dataset.

---

### 4.2 Performance

The system should support fast inference suitable for real-time or near real-time recycling machine operation.

(Need improvement) Inference time should remain sufficiently low to support conveyor-belt-based plastic sorting systems.

---

### 4.3 Scalability (maybe)

The architecture should support:

* Addition of new plastic classes
* Integration of future datasets
* Deployment to embedded or edge-computing systems

---

### 4.4 Reliability (more tests needed)

The system should maintain stable predictions under repeated testing and varying spectra conditions.


---

# Suggested Workflow

Recommended pipeline:

1. Train models on:
   - `FTIR_PLASTIC_c4.csv`
   - `FTIR_PLASTIC_c8.csv`

2. Evaluate generalization on:
   - `openspecy_polymer_dataset.csv`

This helps test model robustness across:
- instruments
- preprocessing pipelines
- spectral variability
- real-world samples

---

# Notes

- Spectral preprocessing may still be required before training.
- Recommended preprocessing:
  - normalization
  - baseline correction
  - interpolation
  - smoothing
  - noise filtering

- OpenSpecy data contains spectra from multiple sources and instruments, which may introduce distribution shifts.

---

# License

Please refer to the original dataset and OpenSpecy licenses for redistribution and usage terms.
