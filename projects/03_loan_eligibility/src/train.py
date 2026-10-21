"""Explore the local loan dataset and compare classifiers without data leakage.

Run from any directory: python path/to/03_loan_eligibility/src/train.py
For the guided explanation, open notebooks/01_loan_eligibility.ipynb.
"""

from importlib.metadata import version
from pathlib import Path
import hashlib
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # Save figures without opening desktop windows.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "raw" / "loan_prediction.csv"
SEED = 42
NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"
]
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"
]
BINARY_FEATURES = ["Credit_History"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
LABEL_NAMES = ["Not Approved", "Approved"]


def load_data(path=DATA_PATH):
    """Validate the expected schema; never invent missing target labels."""
    df = pd.read_csv(path)
    required = set(FEATURES + ["Loan_ID", "Loan_Status"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df["Loan_Status"].isna().any():
        raise ValueError("Training requires nonempty data and no missing Loan_Status.")
    if set(df["Loan_Status"].unique()) != {"Y", "N"}:
        raise ValueError("Loan_Status must contain both Y and N, with no other labels.")
    if df["Loan_ID"].isna().any() or df["Loan_ID"].duplicated().any():
        raise ValueError("Loan_ID must be present and unique to avoid applicant overlap.")
    for column in NUMERIC_FEATURES + BINARY_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"{column} must be numeric.")
        observed = df[column].dropna()
        if (observed < 0).any() or not np.isfinite(observed).all():
            raise ValueError(f"{column} contains negative or nonfinite values.")
    if not set(df["Credit_History"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Credit_History must contain 0, 1, or missing values.")
    return df


def build_preprocessor():
    """Return fresh transformations that learn only inside fit / each CV fold."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
        ("credit", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
    ], remainder="drop")


def build_candidates():
    """Fixed teaching examples, chosen before evaluating the held-out test set."""
    classifiers = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                               random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                                min_samples_leaf=5,
                                                random_state=SEED, n_jobs=1),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])
        for name, model in classifiers.items()
    }


def compare_models(candidates, X_train, y_train):
    """Select by mean CV balanced accuracy, never by the test-set score."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, pipeline in candidates.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv,
                                scoring=scoring, error_score="raise", n_jobs=1)
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)
    # Stable sorting gives deterministic ordering if scores tie.
    return pd.DataFrame(rows).sort_values(
        "balanced_accuracy_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)


def evaluate(model, X_test, y_test):
    """Approved (1) is the positive class for precision, recall, F1 and ROC AUC."""
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "precision_approved": precision_score(y_test, predicted, zero_division=0),
        "recall_approved": recall_score(y_test, predicted, zero_division=0),
        "f1_approved": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }
    return metrics, predicted, probabilities


def save_exploration(df, reports_dir):
    """Descriptive exploration only: these summaries never fit the model."""
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_missing": df.notna().sum(),
        "missing": df.isna().sum(),
        "missing_percent": df.isna().mean().mul(100),
        "unique_non_missing": df.nunique(),
    })
    overview.to_csv(reports_dir / "data_overview.csv", index_label="feature")
    df.describe(include="all").to_csv(reports_dir / "descriptive_statistics.csv")
    df["Loan_Status"].value_counts().rename_axis("Loan_Status").to_csv(
        reports_dir / "target_counts.csv", header=["count"]
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["Loan_Status"].value_counts().reindex(["N", "Y"])
    axes[0].bar(LABEL_NAMES, counts, color=["#cf6660", "#27807c"])
    axes[0].set(title="Recorded loan decisions", ylabel="Applications")
    for i, count in enumerate(counts):
        axes[0].text(i, count + 5, f"{count} ({count / len(df):.1%})", ha="center")
    overview.loc[overview["missing"] > 0, "missing"].sort_values().plot.barh(ax=axes[1])
    axes[1].set(title="Missing values by feature", xlabel="Missing applications")
    fig.tight_layout()
    fig.savefig(figures_dir / "01_target_and_missing.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, column in zip(axes.flat, NUMERIC_FEATURES):
        ax.hist(df[column].dropna(), bins=25, color="#27807c", edgecolor="white")
        ax.set(title=column, xlabel="Value in supplied data units", ylabel="Applications")
    fig.suptitle("Numeric distributions: inspect skew and unusual values")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_numeric_distributions.png", dpi=150)
    plt.close(fig)

    # Include missing credit history explicitly instead of silently dropping it.
    credit = df["Credit_History"].map({0: "0", 1: "1"}).fillna("Missing")
    credit_counts = pd.crosstab(credit, df["Loan_Status"]).reindex(columns=["N", "Y"])
    credit_counts.to_csv(reports_dir / "credit_history_counts.csv")
    credit_rates = credit_counts.div(credit_counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    credit_rates.rename(columns={"N": "Not Approved", "Y": "Approved"}).plot.bar(
        stacked=True, ax=ax, color=["#cf6660", "#27807c"], rot=0
    )
    ax.set(title="Approval proportions by recorded credit history (association only)",
           xlabel="Credit_History", ylabel="Proportion", ylim=(0, 1.2))
    for i, total in enumerate(credit_counts.sum(axis=1)):
        ax.text(i, 1.02, f"n={total}", ha="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_credit_history.png", dpi=150)
    plt.close(fig)
    return overview


def main():
    reports_dir = PROJECT_DIR / "reports"
    models_dir = PROJECT_DIR / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {df.shape[0]} applications and {df.shape[1]} columns.")
    print("Target labels:", df["Loan_Status"].value_counts().to_dict())
    print("Exact duplicate rows:", int(df.duplicated().sum()))
    overview = save_exploration(df, reports_dir)
    print("\nMissing values:\n", overview.loc[overview["missing"] > 0, ["missing", "missing_percent"]])

    X = df[FEATURES]  # Neither the target nor the unique identifier is a predictor.
    y = df["Loan_Status"].map({"N": 0, "Y": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    candidates = build_candidates()
    cv_results = compare_models(candidates, X_train, y_train)
    cv_results.to_csv(reports_dir / "cv_results.csv", index=False)
    best_name = cv_results.iloc[0]["model"]
    best_model = clone(candidates[best_name]).fit(X_train, y_train)
    metrics, predicted, probabilities = evaluate(best_model, X_test, y_test)

    # Fit and score a simple reference on the same partition, without using it to select.
    baseline = clone(candidates["Majority baseline"]).fit(X_train, y_train)
    baseline_metrics, _, _ = evaluate(baseline, X_test, y_test)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    print("\nTraining-only cross-validation:\n", cv_results.to_string(index=False))
    print(f"\nSelected: {best_name}; train={len(X_train)}, test={len(X_test)}")
    print("Held-out metrics:", json.dumps(metrics, indent=2))
    report = classification_report(y_test, predicted, labels=[0, 1],
                                   target_names=LABEL_NAMES, zero_division=0)
    print(report)
    (reports_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    predictions = pd.DataFrame({
        "Loan_ID": df.loc[X_test.index, "Loan_ID"],
        "actual": y_test.map({0: "Not Approved", 1: "Approved"}),
        "predicted": pd.Series(predicted, index=X_test.index).map(
            {0: "Not Approved", 1: "Approved"}),
        "probability_approved": probabilities,
    })
    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    manifest = pd.DataFrame({"Loan_ID": df["Loan_ID"], "split": "train"})
    manifest.loc[X_test.index, "split"] = "test"
    manifest.to_csv(reports_dir / "split_manifest.csv", index=False)
    joblib.dump(best_model, models_dir / "loan_eligibility_pipeline.joblib")

    metadata = {
        "selected_model": best_name, "selection_metric": "mean 5-fold CV balanced accuracy",
        "seed": SEED, "train_rows": len(X_train), "test_rows": len(X_test),
        "label_mapping": {"N": 0, "Y": 1}, "positive_class": "Approved",
        "features": FEATURES, "test_metrics": metrics, "baseline_test_metrics": baseline_metrics,
        "confusion_matrix_rows_actual_columns_predicted_N_then_Y": matrix.tolist(),
        "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
    }
    (reports_dir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, labels=[0, 1],
        display_labels=LABEL_NAMES, cmap="Blues", colorbar=False, ax=axes[0])
    axes[0].set_title(f"Held-out confusion matrix: {best_name}")
    RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
    axes[1].legend(loc="lower right")
    axes[1].set_title("Held-out ROC curve")
    fig.tight_layout()
    fig.savefig(reports_dir / "figures" / "04_model_evaluation.png", dpi=150)
    plt.close(fig)

    tn, fp, fn, tp = matrix.ravel()
    missing_rows = int(df.isna().any(axis=1).sum())
    summary = f"""# Results from the supplied dataset

Generated by `src/train.py`. See the project README and notebook for the teaching walkthrough.

- Applications: **{len(df)}**; columns: **{df.shape[1]}** (11 predictors, ID, target).
- Approved: **{int(y.sum())} ({y.mean():.2%})**; not approved: **{int((y == 0).sum())} ({(y == 0).mean():.2%})**.
- Missing cells: **{int(df.isna().sum().sum())}** in **{missing_rows}** applications; exact duplicate rows: **{int(df.duplicated().sum())}**.
- Split: **{len(X_train)} training / {len(X_test)} test**, stratified, random seed **{SEED}**.
- Selected model: **{best_name}**, using mean balanced accuracy across five training-only folds.
- CV balanced accuracy: **{cv_results.iloc[0]['balanced_accuracy_mean']:.4f}**, fold standard deviation **{cv_results.iloc[0]['balanced_accuracy_std']:.4f}** (not a confidence interval).
- Test accuracy: **{metrics['accuracy']:.4f}**, versus majority baseline **{baseline_metrics['accuracy']:.4f}**.
- Test balanced accuracy: **{metrics['balanced_accuracy']:.4f}**, versus baseline **{baseline_metrics['balanced_accuracy']:.4f}**.
- Approved-class precision: **{metrics['precision_approved']:.4f}**; recall: **{metrics['recall_approved']:.4f}**; F1: **{metrics['f1_approved']:.4f}**.
- Test ROC AUC: **{metrics['roc_auc']:.4f}**.

## Confusion matrix

Rows are actual labels; columns are predicted labels. Approved is positive.

| Actual / predicted | Not Approved | Approved |
| --- | ---: | ---: |
| Not Approved | {tn} | {fp} |
| Approved | {fn} | {tp} |

The model correctly classified **{tn + tp} of {len(y_test)}** held-out applications.
It predicted Approved for **{fp}** records actually marked Not Approved, and
Not Approved for **{fn}** records actually marked Approved. These are errors against
historical decisions, not observed repayment outcomes.

## Exploration

![Target distribution and missing values](figures/01_target_and_missing.png)
![Numeric distributions](figures/02_numeric_distributions.png)
![Credit history and status](figures/03_credit_history.png)
![Held-out evaluation](figures/04_model_evaluation.png)

Income and loan amount have long right tails. ApplicantIncome has a median of
{df['ApplicantIncome'].median():g} and a maximum of {df['ApplicantIncome'].max():g}.
Keep large values unless there is evidence they are erroneous. Credit_History is
associated with status in these records; this is not evidence of causation.

## Limits

This is a small teaching dataset of historical approval decisions. Its source,
license, sampling method, dates, units, and exact credit-history definition have
not been supplied. Test performance is from one split and can vary on new data.
Demographic fields can reproduce historical bias. This exercise does not establish
repayment risk, fairness, probability calibration, or suitability for real lending.
The saved pipeline is fitted on the training partition only; the test set remains
an evaluation set. No model or threshold was chosen using test scores.
"""
    (reports_dir / "findings.md").write_text(summary, encoding="utf-8")
    print(f"\nSaved findings, tables and plots to {reports_dir}")
    print(f"Saved fitted preprocessing + classifier to {models_dir}")


if __name__ == "__main__":
    main()
