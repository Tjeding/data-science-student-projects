# Analyzing and Visualizing Movie Ratings

Clean movie ratings, summarize them, compare movies and genres, and present findings in a simple dashboard.

## Status

Scaffold only; app/ is reserved for the dashboard.

## Data

Dataset to be selected from a public movie ratings source; record its rating scale, source, license, and genre format.

## Planned work

- [ ] Clean missing ratings, duplicates, and genre fields.
- [ ] Calculate mean, median, and mode.
- [ ] Plot rating histograms and box plots.
- [ ] Identify top movies and genres with a documented minimum rating-count rule.
- [ ] Build a dashboard with filters and key findings in app/.

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
