# YouTube Data Dashboard with Streamlit

Build an interactive dashboard for video performance, engagement, and subscriber trends.

## Status

Scaffold only; no API calls, credentials, or dashboard implementation yet.

## Data

Plan public video metadata through the YouTube Data API. Verify authenticated analytics access for watch time and channel growth when implementing.

## Planned work

- [ ] Define available metrics, API access, and refresh cadence.
- [ ] Fetch and cache timestamped records; keep credentials outside Git.
- [ ] Prepare video and channel summaries.
- [ ] Build Streamlit filters, top-video bar charts, and trend charts in app/.
- [ ] Document missing metrics and data freshness.

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
