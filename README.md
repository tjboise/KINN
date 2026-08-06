# KINN: Knowledge-Informed Neural Network for Concrete Compressive Strength Prediction

This repository contains the code for the paper:

> **Knowledge-Informed Neural Network for Concrete Compressive Strength Prediction**
> Tianjie Zhang et al.

---

## Overview

We propose a **Knowledge-Informed Neural Network (KINN)** that embeds Yeh's empirical equation directly into the neural network loss function. By combining data-driven learning with domain physics, KINN achieves more reliable predictions than standard machine learning models, particularly for extrapolation scenarios.

**Hybrid loss function:**

$$\mathcal{L} = (1 - \lambda)\,\mathcal{L}_\text{MSE} + \lambda\,\mathcal{L}_\text{physics}$$

where the physics residual penalizes deviation from Yeh's equation:

$$f_c = (a \cdot \ln(\text{AGE}) + b)\cdot(e \cdot \text{AGE}^d)^{-w/b}$$

---

## Dataset

- **Source:** Port Authority of New York and New Jersey
- **Size:** 6,784 concrete mix designs
- **Target:** Compressive strength $f_c$ (MPa)
- **Features (21):** AGE, PC, PC\_TYPE, FA, SS, SF, FAGG, CAGG, WATER, AEA, WR\_HR, WR, ACC, VOID, w/b, b/a, CAGG%, FAGG%, FA%, SS%, SF%

---

## Models

| Model | Type | Script |
|-------|------|--------|
| **KINN** | Physics-informed neural network | `train_kinn_100times.py` |
| **ANN** | Standard neural network (baseline) | `train_ann_100times.py` |
| **XGBoost** | Gradient boosting | `train_xgb_100times.py` |
| **Random Forest** | Ensemble | `train_rf_100times.py` |
| **SVR** | Support vector regression | `train_svr_100times.py` |
| **KNN** | K-nearest neighbors | `train_knn_100times.py` |
| **Yeh Equation** | Empirical baseline | `empirical_equation.py` |

Each model is trained 100 times to assess robustness. XGBoost, RF, SVR, and KNN include an automatic hyperparameter search step (RandomizedSearchCV or GridSearchCV) before the 100-run loop.

---

## Repository Structure

```
├── model/
│   ├── scripts/                  # Clean .py scripts (main code)
│   │   ├── models.py             # Shared model definitions & constants
│   │   ├── train_kinn_100times.py
│   │   ├── train_ann_100times.py
│   │   ├── train_xgb_100times.py
│   │   ├── train_rf_100times.py
│   │   ├── train_svr_100times.py
│   │   ├── train_knn_100times.py
│   │   ├── empirical_equation.py
│   │   ├── shap_analysis.py          # SHAP DeepExplainer for KINN & ANN
│   │   ├── pdp_plots.py              # Partial Dependence Plots from CSV
│   │   ├── sensitivity_analysis.py   # Lambda sensitivity analysis
│   │   ├── ablation_loss_variants.py # Ablation: 3 knowledge loss designs × 100 runs
│   │   └── grouped_validation.py     # 5-fold GroupKFold by mix design
│   └── *.ipynb                   # Original Jupyter notebooks
└── README.md
```

---

## Results

### Random Split Evaluation (85% train / 15% test, 100 independent runs)

Each model is trained 100 times with different random weight initialisations on a fixed 85/15 data partition.

| Model | Mean Test R² | Std | RMSE (MPa) |
|-------|-------------|-----|------------|
| XGBoost | **0.777** | 0.018 | — |
| Random Forest | 0.727 | 0.019 | — |
| **KINN (ours)** | 0.701 | 0.070 | 7.37 |
| SVR | 0.684 | 0.070 | — |
| ANN | 0.666 | 0.068 | — |

### Grouped Cross-Validation (5-fold, by mix design)

To assess generalisation to unseen mix designs, a 5-fold GroupKFold is applied so that no mix design appears in both training and test folds. The dataset contains 2,271 unique mix designs (6,753 records at 7/28/56-day curing ages); with a random split, 91% of test-set mixes also appear in training at a different age. The grouped CV eliminates this overlap entirely.

| Model | Mean Test R² | Std |
|-------|-------------|-----|
| XGBoost | **0.674** | 0.015 |
| Random Forest | 0.669 | 0.014 |
| ANN | 0.614 | 0.018 |
| KNN | 0.596 | 0.028 |
| KINN (ours) | 0.447 | 0.060 |
| SVR | 0.400 | 0.032 |

The larger performance drop for KINN in the grouped setting reflects that the embedded Yeh physics prior is calibrated on training-mix data; when test mixes have substantially different compositional profiles, the prior can mislead the network. KINN is therefore most appropriate for predicting strength at unobserved ages within the compositional range of the training database (interpolation), not for extrapolation to fundamentally new mix designs.

### Ablation Study: Knowledge Loss Design (λ=0.5, 100 runs)

| Loss Variant | Mean Test R² | Std |
|---|---|---|
| **A: Normalized L1 (proposed)** | **0.709** | 0.012 |
| B: Unnormalized L1 | 0.450 | 0.031 |
| C: Static MSE (conventional) | ≈ −10¹² | — |

Without the adaptive normalization factor, the physics residuals (in physical units) overwhelm the data loss by several orders of magnitude, causing training instability. See `ablation_loss_variants.py` for details.

---

## Environment Setup

```bash
conda create -n concrete_strength python=3.10 -y
conda activate concrete_strength
conda install numpy pandas scipy scikit-learn matplotlib -y
pip install torch torchvision torchaudio
pip install xgboost shap openpyxl joblib
conda install jupyter notebook ipykernel -y
python -m ipykernel install --user --name concrete_strength --display-name "concrete_strength"
```

> For GPU support, replace the PyTorch install with:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

---

## Usage

All scripts are run from the `model/` directory with the data file `updated_fc_predictions.xlsx` present.

**Train KINN (100 runs):**
```bash
cd model
python scripts/train_kinn_100times.py
```

**Train all baseline models:**
```bash
python scripts/train_ann_100times.py
python scripts/train_xgb_100times.py
python scripts/train_rf_100times.py
python scripts/train_svr_100times.py
python scripts/train_knn_100times.py
python scripts/empirical_equation.py
```

**SHAP analysis:**
```bash
python scripts/shap_analysis.py
```

**Partial Dependence Plots** (requires `pdp_results_7/28/56.csv`):
```bash
python scripts/pdp_plots.py
```

**Lambda sensitivity analysis:**
```bash
python scripts/sensitivity_analysis.py
```

**Ablation study (knowledge loss variants):**
```bash
python scripts/ablation_loss_variants.py
```

**Grouped cross-validation (mix-design-level generalisation):**
```bash
python scripts/grouped_validation.py
```

---

## Key Outputs

| Script | Output files |
|--------|-------------|
| `train_kinn_100times.py` | `model_evaluation_metrics_with_scaled.xlsx` |
| `train_ann_100times.py` | `ANN_best_trial_results.xlsx` |
| `train_xgb_100times.py` | `XGBoost_best_trial_results.xlsx` |
| `train_rf_100times.py` | `RandomForest_best_trial_results.xlsx` |
| `train_svr_100times.py` | `SVR_best_trial_results.xlsx` |
| `train_knn_100times.py` | `KNN_best_trial_results.xlsx` |
| `empirical_equation.py` | `empirical_equation_results.xlsx` |
| `shap_analysis.py` | `KINN_summary_plot.png`, `KINN_bar_plot.png`, `ANN_shap_summary_plot.png`, `shap_mean_absolute_bar_plot_ranked.png` |
| `sensitivity_analysis.py` | `KINN_hybrid_results.xlsx`, `hybrid_sensitivity_plot.png` |
| `pdp_plots.py` | `PDP_*.png` |
| `ablation_loss_variants.py` | `ablation_results.xlsx`, `ablation_summary.png` |
| `grouped_validation.py` | `grouped_cv_results.xlsx`, `grouped_cv_summary.xlsx` |
