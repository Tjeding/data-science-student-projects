"""Explore scikit-learn's bundled Iris data; save tables, figures and findings.

Run from any directory: python path/to/05_iris_data_exploration/src/explore.py
Charts open in interactive windows by default. Add --no-show to save files only.
The guided notebook explains each step. No network access is needed.
"""

from importlib.metadata import version
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

PROJECT_DIR = Path(__file__).resolve().parents[1]
FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
SPECIES = ["setosa", "versicolor", "virginica"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "s", "^"]


def load_data():
    """Return four numeric measurements plus a human-readable species column."""
    iris = load_iris(as_frame=True)
    df = iris.data.copy()
    df["species"] = pd.Categorical.from_codes(iris.target.to_numpy(), iris.target_names)
    return df


def prepare_data(raw):
    """Preserve observations; median-fill missing measurements only if needed.

    This is descriptive exploration, not preprocessing for a predictive model.
    If modeling later, fit imputation on training data only.
    """
    df = raw.copy(deep=True)
    if df.empty or df["species"].isna().any():
        raise ValueError("Exploration requires rows with known species; do not guess labels.")
    if not set(df["species"]).issubset(SPECIES):
        raise ValueError("Unexpected species label.")
    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f"{feature} must be numeric.")
        observed = df[feature].dropna()
        if observed.empty:
            raise ValueError(f"Cannot estimate a median for all-missing {feature}.")
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError(f"{feature} contains nonpositive or nonfinite measurements; review them.")
    missing = df[FEATURES].isna().sum()
    if missing.any():
        df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df, missing


def species_legend():
    return [Line2D([], [], color=color, marker=marker, linestyle="none", label=name)
            for name, color, marker in zip(SPECIES, COLORS, MARKERS)]


def plot_histograms(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feature in zip(axes.flat, FEATURES):
        bins = np.histogram_bin_edges(df[feature], bins=12)
        ax.hist(df[feature], bins=bins, color="#d8dee4", edgecolor="white", label="All species")
        for name, color in zip(SPECIES, COLORS):
            ax.hist(df.loc[df["species"].eq(name), feature], bins=bins,
                    histtype="step", linewidth=2, color=color, label=name)
        ax.set(xlabel=feature, ylabel="Flowers", title=feature.replace(" (cm)", ""))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions: all flowers and each species (shared bins)")
    fig.tight_layout()
    return fig


def plot_boxplots(df):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feature in zip(axes.flat, FEATURES):
        groups = [df.loc[df["species"].eq(name), feature] for name in SPECIES]
        boxes = ax.boxplot(groups, tick_labels=SPECIES, patch_artist=True,
                           medianprops={"color": "black", "linewidth": 1.5})
        for box, color in zip(boxes["boxes"], COLORS):
            box.set_facecolor(color)
            box.set_alpha(0.5)
        ax.set(ylabel=feature, title=feature.replace(" (cm)", ""))
    fig.suptitle("Compare the median, spread and possible outliers by species")
    fig.tight_layout()
    return fig


def plot_scatterplots(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (x, y) in zip(axes, [(FEATURES[0], FEATURES[1]), (FEATURES[2], FEATURES[3])]):
        for name, color, marker in zip(SPECIES, COLORS, MARKERS):
            group = df.loc[df["species"].eq(name)]
            ax.scatter(group[x], group[y], color=color, marker=marker, alpha=0.75,
                       s=35, label=name)
        ax.set(xlabel=x, ylabel=y)
    axes[0].set_title("Sepal measurements: substantial overlap")
    axes[1].set_title("Petal measurements: setosa stands apart")
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_pairplot(df):
    """A Matplotlib pair plot: histograms on the diagonal, scatterplots elsewhere."""
    fig, axes = plt.subplots(4, 4, figsize=(12, 11))
    for row, y in enumerate(FEATURES):
        for column, x in enumerate(FEATURES):
            ax = axes[row, column]
            if row == column:
                bins = np.histogram_bin_edges(df[x], bins=10)
                for name, color in zip(SPECIES, COLORS):
                    ax.hist(df.loc[df["species"].eq(name), x], bins=bins,
                            color=color, histtype="step", linewidth=1.5)
                ax.text(0.05, 0.90, "Count", transform=ax.transAxes, fontsize=8)
            else:
                for name, color, marker in zip(SPECIES, COLORS, MARKERS):
                    group = df.loc[df["species"].eq(name)]
                    ax.scatter(group[x], group[y], color=color, marker=marker, s=12, alpha=0.6)
            if row == 3:
                ax.set_xlabel(x)
            if column == 0:
                ax.set_ylabel("Count" if row == column else y)
            ax.tick_params(labelsize=8)
    fig.legend(handles=species_legend(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Pair plot: distributions and every pair of measurements", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_correlations(df):
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    chart = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [name.replace(" (cm)", "") for name in FEATURES]
    ax.set_xticks(range(4), names, rotation=25, ha="right")
    ax.set_yticks(range(4), names)
    for row in range(4):
        for column in range(4):
            value = corr.iloc[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.6 else "black")
    fig.colorbar(chart, ax=ax, label="Pearson correlation (-1 to 1)")
    ax.set_title("Feature correlations across all 150 flowers")
    fig.tight_layout()
    return fig


def main(show=False):
    raw = load_data()
    df, imputed = prepare_data(raw)
    reports = PROJECT_DIR / "reports"
    figures = reports / "figures"
    processed = PROJECT_DIR / "data/processed"
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame({
        "dtype": raw.dtypes.astype(str), "non_missing": raw.notna().sum(),
        "missing": raw.isna().sum(), "missing_percent": raw.isna().mean() * 100,
        "unique_values": raw.nunique(),
    })
    overview.to_csv(reports / "data_overview.csv", index_label="column")
    df[FEATURES].describe().T.to_csv(reports / "descriptive_statistics.csv", index_label="feature")
    df.groupby("species", observed=True)[FEATURES].agg(
        ["mean", "median", "std", "min", "max"]
    ).to_csv(reports / "statistics_by_species.csv")
    counts = df["species"].value_counts(sort=False)
    counts.to_csv(reports / "species_counts.csv", index_label="species", header=["count"])
    duplicates = raw.loc[raw.duplicated(keep=False)]
    duplicates.to_csv(reports / "repeated_measurements.csv", index_label="source_row_index")
    df[FEATURES].corr().to_csv(reports / "feature_correlations.csv", index_label="feature")
    dataset_csv = df.to_csv(index=False, lineterminator="\n")
    (processed / "iris_explored.csv").write_bytes(dataset_csv.encode("utf-8"))
    figure_functions = [
        (plot_histograms, "01_feature_histograms.png"),
        (plot_boxplots, "02_boxplots_by_species.png"),
        (plot_scatterplots, "03_scatterplots.png"),
        (plot_pairplot, "04_pair_plot.png"),
        (plot_correlations, "05_correlations.png"),
    ]
    for plot, filename in figure_functions:
        fig = plot(df)
        fig.savefig(figures / filename, dpi=150)
        if not show:
            plt.close(fig)
    summary = {
        "source": "sklearn.datasets.load_iris(as_frame=True)",
        "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html",
        "rows": len(df), "columns": list(df.columns),
        "feature_units": "centimeters", "species_counts": {k: int(v) for k, v in counts.items()},
        "missing_cells_before": int(raw.isna().sum().sum()),
        "imputed_feature_cells": int(imputed.sum()),
        "duplicate_rows_beyond_first": int(raw.duplicated().sum()),
        "rows_in_repeated_groups": len(duplicates),
        "duplicate_policy": "Retained: equal rounded measurements do not establish a recording error.",
        "exported_csv_sha256": hashlib.sha256(dataset_csv.encode("utf-8")).hexdigest(),
        "versions": {name: version(name) for name in ["numpy", "pandas", "matplotlib", "scikit-learn"]},
    }
    (reports / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stats_rows = "\n".join(
        f"| {name} | {df[name].min():.1f} | {df[name].max():.1f} | {df[name].mean():.3f} | {df[name].median():.2f} |"
        for name in FEATURES
    )
    petal_corr = df[FEATURES].corr().loc[FEATURES[2], FEATURES[3]]
    findings = f"""# Task 5: Iris exploration findings

Generated by `src/explore.py` from scikit-learn's bundled Iris dataset.

## Structure and quality

- **{len(df)} rows**, **4 numeric features**, and a categorical `species` label.
- Species counts: {', '.join(f'{name} = {count}' for name, count in counts.items())}.
- All four measurements are in **centimeters**.
- Missing cells: **{summary['missing_cells_before']}**; imputed feature cells: **{int(imputed.sum())}**.
- Repeated rows beyond the first occurrence: **{summary['duplicate_rows_beyond_first']}**
  ({len(duplicates)} rows in the repeated group). Retained; matching measurements
  at the recorded precision do not prove a duplicated observation.

The supplied data needs no missing-value imputation. No observations were removed
or scaled. The optional missing-feature branch uses medians for descriptive
exploration only and rejects missing species or entirely missing features.

## Feature distributions

| Feature | Minimum | Maximum | Mean | Median |
| --- | ---: | ---: | ---: | ---: |
{stats_rows}

![Feature distributions](figures/01_feature_histograms.png)

Each gray histogram shows all flowers; colored outlines show species using the
same bin edges. Sepal length spans 4.3–7.9 cm with overlapping species; sepal width
spans 2.0–4.4 cm and also overlaps strongly. Petal length and width show a distinct
small-petal setosa group. The combined distribution mixes three species, so
multiple peaks need not indicate an error. Bin choices affect histogram shape.

![Box plots by species](figures/02_boxplots_by_species.png)

Boxes show the middle 50% of measurements; the line is the median. Whiskers extend
to the most extreme observations within 1.5 interquartile ranges of the box.
Points beyond them are possible outliers, not automatically errors. Setosa has
the smallest petals; virginica has the largest average petal measurements;
versicolor is intermediate. Sepal distributions overlap more.

## Relationships

![Sepal and petal scatter plots](figures/03_scatterplots.png)
![Pair plot](figures/04_pair_plot.png)

Each scatter point represents a flower; identical measurements may overlap.
Petal measurements separate setosa clearly in this sample, while versicolor and
virginica overlap. A pair plot shows every pair of measurements, with histograms
on the diagonal. Its diagonal y-axis is a count, not a measurement.

![Feature correlations](figures/05_correlations.png)

Petal length and petal width have a strong overall positive Pearson correlation
(**{petal_corr:.3f}**). This is association, not causation. Pooling species can
strengthen or hide relationships within species. The species labels are
categories and are excluded from numerical correlations.

## Conclusion and limits

This exploration meets the task: load the data, inspect its structure, check
missingness and repeated rows, summarize each feature and visualize distributions.
No missing-value treatment was necessary. Petal measurements show clearer
species differences than sepal measurements in this dataset. This small,
balanced teaching sample does not establish population proportions or guarantee
classification performance. No predictive model was trained or evaluated.

## Sources and reproduction

Use the [scikit-learn loader documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
and [data provenance record](../data/README.md). The loader documents corrections
relative to the original UCI copy. The historical dataset is credited to
R. A. Fisher (1936); the [UCI record](https://www.archive.ics.uci.edu/dataset/53/iris)
lists CC BY 4.0. The local analysis uses scikit-learn's packaged version.
`reports/summary.json` records dependency versions and the exported CSV fingerprint.
The script resolves paths relative to itself and needs no network connection.
"""
    (reports / "findings.md").write_text(findings, encoding="utf-8")
    print(f"Loaded {len(df)} flowers, {len(FEATURES)} numeric features and 1 species label.")
    print("Species counts:", counts.to_dict())
    print(f"Missing cells: {summary['missing_cells_before']}; filled: {int(imputed.sum())}.")
    print(f"Repeated rows beyond the first: {summary['duplicate_rows_beyond_first']} (retained).")
    print(df[FEATURES].describe().round(3).to_string())
    print(f"Saved tables, five charts and findings to {reports}")
    if show:
        print("Opening all five charts. Close the plot windows to finish the script.")
        plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", action="store_true", dest="show",
                         help="Open plot windows as well as saving charts (the default).")
    display.add_argument("--no-show", action="store_false", dest="show",
                         help="Save charts and reports without opening plot windows.")
    parser.set_defaults(show=True)
    args = parser.parse_args()
    # Tk provides interactive windows; Agg supports saving without a desktop.
    # Configure this only for the script so importing it preserves notebook plots.
    matplotlib.use("TkAgg" if args.show else "Agg")
    main(show=args.show)
