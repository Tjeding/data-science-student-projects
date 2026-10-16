# COVID-19 Data Analysis

Analyze time trends in reported cases, recoveries, deaths, and vaccination rates where available.

## Status

Scaffold only; live ingestion will be designed later.

## Data

Select and document a public source, retrieval time, geographic coverage, reporting cadence, and field definitions. Freshness depends on the source.

## Planned work

- [ ] Collect a dated snapshot and record provenance.
- [ ] Clean dates, missing reports, duplicates, and geographic identifiers.
- [ ] Distinguish daily counts from cumulative totals and population-adjusted rates.
- [ ] Visualize trends and explore associations without assuming causation.

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
