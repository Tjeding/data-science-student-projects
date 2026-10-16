# Linear Regression on Housing Prices

Preprocess housing data and predict prices using multiple linear regression with three predictors: average rooms (`rm`), `lstat`, and pupil-teacher ratio (`ptratio`). A one-feature baseline remains planned.

## Status

Existing work: src/train.py (formerly Task2.py). It splits data 80/20 with random seed 42 before fitting the scaler on training data only. The test set and sample prediction use that fitted scaler.

## Run

From the repository root after installing the common dependencies:

```powershell
.\.venv\Scripts\python.exe projects/02_housing_linear_regression/src/train.py
```

The script reports MAE, MSE, RMSE, and R-squared, then predicts a sample house value. It writes `reports/regression_coefficients.csv` and `data/processed/cleaned_boston_housing.csv`. Coefficients describe a one-standard-deviation feature increase, holding the other predictors fixed, because inputs are standardized. The fitted model is not saved yet.

## Data

Existing BostonHousing.csv is preserved in data/raw/. Source and license still need to be documented.

## Planned work

- [ ] Explore features and select a target and baseline predictor.
- [x] Split data before fitting scalers; fit preprocessing on training data only.
- [ ] Train a one-feature baseline, then optionally compare multiple regression.
- [x] Evaluate MAE, RMSE, and R-squared.
- [ ] Plot predictions and residuals.

## Layout

| Folder | Purpose |
| --- | --- |
| `data/raw/` | Original input files; keep unchanged. |
| `data/interim/` | Intermediate cleaning outputs. |
| `data/processed/` | Analysis-ready or model-ready data. |
| `notebooks/` | Numbered Jupyter notebooks, such as `01_exploration.ipynb`. |
| `src/` | Reusable analysis, preprocessing, and training code. |
| `models/` | Generated model artifacts. |
| `reports/` | Written findings, metrics, and tables. |
| `reports/figures/` | Exported plots. |

Use the environment setup in the [root README](../../README.md). Record project-specific dependencies in a local `requirements.txt` when needed.
