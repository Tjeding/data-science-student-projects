# Model Evaluation

Evaluate a classification model on a held-out test set and explain its errors.

## Status

Scaffold only; evaluation code will be added later.

## Data

Use a model and held-out predictions from the loan, Iris, or Titanic project; document their paths and provenance.

## Planned work

- [ ] Define positive class and keep the test set separate from tuning.
- [ ] Calculate accuracy, precision, recall, and F1.
- [ ] Visualize a confusion matrix and produce a classification report.
- [ ] Discuss class imbalance, error types, and limitations.

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
