# HB Prediction from Near-Infrared Spectral Data

Predicting hemoglobin (HB) concentration in dialysis patients using near-infrared (NIR) absorption spectra and machine learning.

---

## Background

- **Data**: NIR spectrum files (`.txt`) collected during dialysis sessions, matched with clinical HB values from Excel records
- **Patients**: ~82 patients / ~292 matched samples across 4 collection dates
- **Goal**: Predict ClinicHb (g/dL) from spectral features; also classify HB < 10 vs HB ≥ 10

---

## Development Timeline

| Date | Work Done |
|------|-----------|
| 2026-04-25 | Built base SWT regression model with 5 derived features (v540-v500 etc.), patient-level KFold |
| 2026-04-28 | Split into HB < 10 / HB ≥ 10 two-group models; added 4:1 train/test split |
| 2026-05-05 | Switched to 3-channel joint normalization (v540, v560, v577); removed early stopping; added binary classifier, FFT, derivative exploration scripts; pushed to GitHub |

---

## Current Status (2026-05-05)

| Stage | Status |
|-------|--------|
| Data pairing (spectrum ↔ Excel) | Done |
| Feature exploration (derivatives, FFT) | Done |
| Regression model (SWT + 3-channel) | Trained, poor R² — under investigation |
| Binary classification model | Code ready, not yet trained |
| Feature update based on exploration | Pending |

**Key finding from exploration**: 2nd-order derivative (d2) features at 537–540nm show the strongest individual correlation with HB (r ≈ 0.33). Raw SWT values and FFT features are weaker (r ≈ 0.22–0.29).

---

## File Overview

### Data Check

| File | Description |
|------|-------------|
| `check_hb_split.py` | Counts matched samples/patients by cross-referencing spectrum files with Excel. Shows HB < 10 / HB ≥ 10 split and expected 4:1 train/test distribution. |
| `check_hb_split_0505.py` | Updated version of the above (same function, minor fixes). |

---

### Regression Training & Evaluation

| File | Description |
|------|-------------|
| `train_swt_hb_split.py` | Main regression training script. Splits data into **HB < 10** and **HB ≥ 10** groups, trains one model per group. Uses SWT level-3 (db4) features: raw cA3 values at 540 / 560 / 577nm, joint-normalized. Patient-level 4:1 train/test split + 5-fold cross-validation. |
| `train_swt_hb_split_0505.py` | Updated version with configurable `SWT_LEVEL`, optional demographic features (`DEMO_COLS`), and dynamic `BASE_DIR` path detection. |
| `eval_swt_hb_split.py` | Loads the two trained models and evaluates on their respective test sets. Outputs a 4-panel figure per group: regression scatter, sorted prediction, residual plot, normal probability plot. |
| `eval_swt_hb_split_0505.py` | Updated version of eval (matches `train_swt_hb_split_0505.py`). |

**Model outputs**: `swt_hb_low_model.pth`, `swt_hb_high_model.pth`

---

### Binary Classification

| File | Description |
|------|-------------|
| `train_binary_hb_0505.py` | Trains a binary classifier on all data to predict **HB < 10 (label=0)** or **HB ≥ 10 (label=1)**. Uses BCEWithLogitsLoss with positive-class weighting for imbalanced data. Reports Accuracy, AUC-ROC, F1, Precision, Recall, and outputs an ROC curve plot. Same SWT features as regression model. |

**Model output**: `swt_hb_binary_model.pth`

---

### Feature Exploration (run before training to understand data)

| File | Description |
|------|-------------|
| `explore_derivatives_0505.py` | Computes 1st and 2nd order derivatives of SWT cA3 coefficients across the full 500–800nm range. Calculates Pearson correlation of each wavelength's raw / cA / d1 / d2 value with ClinicHb. Outputs a 4-panel correlation overview and a scatter plot of the top-3 most correlated wavelengths for d1 and d2. |
| `explore_fft_ch542_0505.py` | Two FFT analyses at channel 542nm: (A) temporal FFT of repeated measurements within a single file; (B) FFT of the averaged spectrum to extract frequency-domain shape features. Plots correlation of each FFT coefficient with ClinicHb. |
| `explore_swt2_deriv_ch659_0505.py` | Applies SWT level-2 (db4) to the spectrum, computes the 1st derivative of cA2, and evaluates the slope at channel 659nm. Also scans the 600–720nm range to find the wavelength with highest d1 correlation with ClinicHb. |

---

## Feature Summary (from exploration)

| Feature | Best wavelength | Pearson r | Note |
|---------|----------------|-----------|------|
| raw / SWT cA3 | 540–580nm | ~+0.24 | Currently used in training |
| 1st deriv (d1) | 570nm | ~−0.28 | Slightly better than raw |
| **2nd deriv (d2)** | **537nm** | **~+0.33** | **Strongest individual signal** |
| 659nm (any) | 659nm | ~0.04 | Not useful |
| FFT spectrum | coef #26 | ~+0.29 | No gain over raw |

---

## How to Run (on server)

```bash
# 1. Check data distribution
python3 update_0505/check_hb_split_0505.py

# 2. Explore features (run all three, can be parallel)
python3 update_0505/explore_derivatives_0505.py
python3 update_0505/explore_fft_ch542_0505.py
python3 update_0505/explore_swt2_deriv_ch659_0505.py

# 3. Train regression models
python3 update_0505/train_swt_hb_split_0505.py

# 4. Evaluate
python3 update_0505/eval_swt_hb_split_0505.py

# 5. Train binary classifier
python3 update_0505/train_binary_hb_0505.py
```

---

## Notes

- **Do NOT share**: `.txt` spectrum files, `.xlsx` Excel files, `.pth` model files (contain patient session IDs)
- **Safe to share**: all `.py` files in this folder
- `BASE_DIR` is auto-detected as the parent folder of the script — no hardcoded paths
- Chinese font warnings on Linux server are cosmetic (plot labels use English)
