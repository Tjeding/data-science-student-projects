# Titanic Survival Portfolio Project

Build a passenger survival classifier with cleaning, feature engineering, and evaluation.

## Status

Scaffold only; the earlier EDA stays in project 01 and can inform this project.

## Data

Reuse ../01_public_dataset_eda/data/raw/titanic.csv as a starting dataset. Document any later Kaggle train/test files separately.

## Planned work

- [ ] Explore survival patterns and split labeled data.
- [ ] Build preprocessing and passenger feature engineering without target leakage.
- [ ] Train and compare classification models.
- [ ] Evaluate on held-out labeled data and document the workflow.
- [ ] Optionally prepare predictions for an unlabeled competition test set.

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
