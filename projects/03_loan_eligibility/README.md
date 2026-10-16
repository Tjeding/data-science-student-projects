# Loan Eligibility Prediction

Explore loan applications and train a classifier for approved versus not approved status.

## Status

Scaffold only; no dataset or model yet.

## Data

Dataset to be selected; document the target labels, source, license, and feature meanings.

## Planned work

- [ ] Inspect features, target balance, missing values, and duplicates.
- [ ] Split data and build numeric and categorical preprocessing.
- [ ] Compare a baseline classifier with additional algorithms.
- [ ] Evaluate accuracy, precision, recall, F1, and a confusion matrix.

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
