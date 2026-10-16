# Exploratory Data Analysis on a Public Dataset

Clean a public dataset, handle missing values, calculate descriptive statistics, and visualize distributions, correlations, and key patterns.

## Status

Existing work: src/eda.py (formerly Task1.py) and data/processed/titanic_cleaned.csv. The script supports pandas string columns and fills missing ages and embarkation values using direct column assignment. This is a starting script, not a completed notebook.

## Run

From the repository root after installing the common dependencies:

```powershell
.\.venv\Scripts\python.exe projects/01_public_dataset_eda/src/eda.py
```

The script displays plots and saves `data/processed/titanic_cleaned.csv`. Ages use the dataset median and embarkation values use the most frequent port; the mostly missing cabin column is dropped. These whole-dataset summaries are for exploration; a future prediction model must learn imputation values from its training split only.

## Data

Titanic CSV already included in data/raw/titanic.csv. Record its original source and license before publishing derived data.

## Planned work

- [ ] Inspect structure, data types, duplicates, and missing values.
- [ ] Document cleaning decisions and summary statistics.
- [ ] Plot distributions, relationships, and correlations.
- [ ] Write a short report of findings and limitations.

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
