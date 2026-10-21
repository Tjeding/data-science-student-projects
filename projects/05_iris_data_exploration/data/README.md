# Data provenance

The source is the Iris dataset packaged with scikit-learn, loaded using
`sklearn.datasets.load_iris(as_frame=True)`. No separate dataset is downloaded.

- Loader: [official scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
- Historical attribution: R. A. Fisher (1936), Iris dataset; see
  [UCI's record](https://www.archive.ics.uci.edu/dataset/53/iris), DOI 10.24432/C56C76.
- License context: the UCI record lists **CC BY 4.0**. This project uses the
  scikit-learn packaged version, whose loader documents corrections relative to
  the original UCI copy; it is not a fresh download of UCI's CSV.
- Local source version: scikit-learn **1.9.1**, pinned in `../requirements.txt`.
- Inspection date: **2026-10-21 (Africa/Johannesburg)**. No dataset network
  retrieval occurred, so there is no retrieval date for this analysis.
- Shape: 150 samples, four numeric features and a species label; 50 per species.
- Features: sepal length, sepal width, petal length and petal width, all in **cm**.
- Label mapping from the loader: 0 = setosa, 1 = versicolor, 2 = virginica. The
  analysis exports named species as a categorical column, rather than treating
  codes as continuous measurements.

## Preparation and reproducibility

The packaged dataset has **no missing cells**. No values are imputed, scaled or
removed. One row repeats an earlier complete row (source indices 101 and 142).
Both are retained because equal rounded measurements do not prove an accidental
duplicate. Repeated measurements are exported for inspection.

Run `src/explore.py` to load the packaged source and export
`data/processed/iris_explored.csv`. The original source is unchanged.
`reports/summary.json` records the software versions and a SHA-256 fingerprint
of the UTF-8, LF-terminated exported CSV. Numeric medians would be used only if
feature gaps were found; missing species and all-missing features are rejected.

This is a small historical teaching sample with balanced class counts, not
evidence of population proportions. Recorded precision allows measurements to
repeat. Species associations and correlations are descriptive; they do not
establish causation or predictive accuracy. New generated files follow the
repository's existing Git ignore rules.
