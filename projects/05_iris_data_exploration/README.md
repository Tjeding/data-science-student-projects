# Iris Data Exploration

Explore the Iris dataset, handle missing values if present, and visualize each feature.

## Status

Scaffold only; no analysis yet.

## Data

Load Iris through scikit-learn when implementing the project; no download or data copy is needed now.

## Planned work

- [ ] Inspect shape, feature names, types, and species labels.
- [ ] Check missing values and duplicates; document any cleaning.
- [ ] Summarize features overall and by species.
- [ ] Create histograms, box plots, scatter plots, and a pair plot.

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
