"""Evaluate Task 3's frozen loan model on its original test partition.

Run from any working directory: python path/to/04_model_evaluation/src/evaluate.py
No preprocessing or classifier is fitted here. See the notebook for the lesson.
"""

from importlib.metadata import version
from pathlib import Path
import hashlib
import json

import joblib
import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")  # Preserve the inline backend when imported by the lesson.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR.parent / "03_loan_eligibility"
LABELS = [0, 1]
LABEL_NAMES = ["Not Approved", "Approved"]


def load_test_predictions(source_dir=SOURCE_DIR):
    """Check provenance, restore the saved partition, and predict without fit()."""
    data_path = source_dir / "data/raw/loan_prediction.csv"
    metadata_path = source_dir / "reports/metrics.json"
    manifest_path = source_dir / "reports/split_manifest.csv"
    model_path = source_dir / "models/loan_eligibility_pipeline.joblib"
    predictions_path = source_dir / "reports/test_predictions.csv"
    for path in [data_path, metadata_path, manifest_path, model_path, predictions_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Task 3 input: {path}. Run 03_loan_eligibility/src/train.py first."
            )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if fingerprint != metadata["dataset_sha256"]:
        raise ValueError("The raw data differs from Task 3's recorded dataset. Stop and investigate.")
    # Pickled estimators require compatible dependencies. Do this before loading.
    for package, expected in metadata["versions"].items():
        if version(package) != expected:
            raise ValueError(
                f"{package} must match Task 3: expected {expected}, found {version(package)}. "
                "Use Task 3's requirements.txt."
            )
    if metadata["label_mapping"] != {"N": 0, "Y": 1}:
        raise ValueError("Expected the Task 3 mapping N=0 and Y=1.")
    data = pd.read_csv(data_path)
    manifest = pd.read_csv(manifest_path)
    for name, table in [("raw data", data), ("split manifest", manifest)]:
        if table["Loan_ID"].isna().any() or table["Loan_ID"].duplicated().any():
            raise ValueError(f"Missing or duplicate Loan_ID in {name}.")
    if set(data["Loan_ID"]) != set(manifest["Loan_ID"]):
        raise ValueError("The split manifest does not cover exactly the raw applications.")
    if set(manifest["split"]) != {"train", "test"}:
        raise ValueError("The manifest must contain only train and test partitions.")
    partitions = manifest["split"].value_counts()
    if any(partitions[part] != metadata[f"{part}_rows"] for part in ["train", "test"]):
        raise ValueError("Partition counts differ from the Task 3 record.")
    if data["Loan_Status"].isna().any() or set(data["Loan_Status"]) != {"N", "Y"}:
        raise ValueError("Expected complete N/Y target labels.")
    indexed = data.set_index("Loan_ID")
    test_ids = manifest.loc[manifest["split"].eq("test"), "Loan_ID"]
    train_ids = manifest.loc[manifest["split"].eq("train"), "Loan_ID"]
    test = indexed.loc[test_ids]
    y_test = test["Loan_Status"].map(metadata["label_mapping"]).to_numpy()
    if set(y_test) != set(LABELS):
        raise ValueError("Both classes must be present in this evaluation set.")
    # This artifact was created locally by Task 3; only load trusted joblib files.
    model = joblib.load(model_path)
    y_pred = model.predict(test[metadata["features"]])
    if not set(y_pred).issubset(LABELS):
        raise ValueError("The model returned labels other than 0 and 1.")
    positive_column = list(model.classes_).index(1)
    scores = model.predict_proba(test[metadata["features"]])[:, positive_column]
    # Align by application ID rather than assuming the two files share row order.
    saved = pd.read_csv(predictions_path)
    if saved["Loan_ID"].isna().any() or saved["Loan_ID"].duplicated().any():
        raise ValueError("Task 3's predictions contain missing or duplicate IDs.")
    if set(saved["Loan_ID"]) != set(test_ids):
        raise ValueError("Saved predictions do not correspond exactly to the test partition.")
    saved = saved.set_index("Loan_ID").loc[test_ids]
    names = np.array(LABEL_NAMES)
    if not (np.array_equal(saved["actual"], names[y_test])
            and np.array_equal(saved["predicted"], names[y_pred])
            and np.allclose(saved["probability_approved"], scores, rtol=1e-9, atol=1e-12)):
        raise ValueError("The saved model, labels, and Task 3 predictions disagree.")
    y_train = indexed.loc[train_ids, "Loan_Status"].map(metadata["label_mapping"])
    majority_class = int(y_train.mode().iloc[0])  # Determined from training labels only.
    baseline = np.full(len(y_test), majority_class)
    predictions = pd.DataFrame({
        "Loan_ID": test_ids.to_numpy(),
        "actual": names[y_test],
        "predicted": names[y_pred],
        "probability_approved": scores,
        "correct": y_test == y_pred,
    })
    provenance = {
        "source_project": "03_loan_eligibility",
        "selected_model": metadata["selected_model"],
        "selection_metric": metadata["selection_metric"],
        "dataset_sha256": fingerprint,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "split_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_rows": len(train_ids), "test_rows": len(test_ids),
        "positive_class": "Approved (1)",
        "baseline_class": LABEL_NAMES[majority_class],
        "versions": metadata["versions"],
        "task3_predictions_reproduced": True,
    }
    return y_test, y_pred, baseline, predictions, provenance


def calculate_metrics(y_true, y_pred):
    """Precision, recall, and F1 use Approved (1) as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_approved": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_approved": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_approved": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def plot_confusion_matrices(y_true, y_pred):
    """Show counts and per-actual-class proportions with explicit label order."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, normalize, title, fmt in [
        (axes[0], None, "Test predictions: application counts", "d"),
        (axes[1], "true", "Test predictions: proportion within each actual class", ".1%"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=LABELS, display_labels=LABEL_NAMES,
            normalize=normalize, cmap="Blues", colorbar=False,
            values_format=fmt, ax=ax,
        )
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def plot_classification_report(report):
    """Visualize class scores; support counts stay in the row labels."""
    columns = ["precision", "recall", "f1-score"]
    values = np.array([[report[name][metric] for metric in columns] for name in LABEL_NAMES])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    chart = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3), ["Precision", "Recall", "F1 score"])
    ax.set_yticks(range(2), [f"{name} (n={int(report[name]['support'])})" for name in LABEL_NAMES])
    for row in range(2):
        for column in range(3):
            value = values[row, column]
            ax.text(column, row, f"{value:.2%}", ha="center", va="center",
                    color="white" if value > 0.65 else "black")
    ax.set_title("Classification report: performance for each class")
    fig.colorbar(chart, ax=ax, label="Score (0 to 1)")
    fig.tight_layout()
    return fig


def main():
    y_test, y_pred, baseline, predictions, provenance = load_test_predictions()
    metrics = calculate_metrics(y_test, y_pred)
    baseline_metrics = calculate_metrics(y_test, baseline)
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
    report = classification_report(y_test, y_pred, labels=LABELS,
                                   target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    text_report = classification_report(y_test, y_pred, labels=LABELS,
                                        target_names=LABEL_NAMES, digits=4, zero_division=0)
    comparison = pd.DataFrame([metrics, baseline_metrics],
                              index=[provenance["selected_model"], "Majority baseline"])
    reports = PROJECT_DIR / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    predictions.loc[~predictions["correct"]].to_csv(reports / "misclassified_applications.csv", index=False)
    comparison.to_csv(reports / "metrics_comparison.csv", index_label="model")
    pd.DataFrame({name: report[name] for name in
                  [*LABEL_NAMES, "macro avg", "weighted avg"]}).T.to_csv(
        reports / "classification_report.csv", index_label="class_or_average")
    (reports / "classification_report.txt").write_text(text_report, encoding="utf-8")
    pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        reports / "confusion_matrix.csv", index_label="actual / predicted")
    metadata = {**provenance, "test_metrics": metrics, "baseline_test_metrics": baseline_metrics,
                "confusion_matrix_rows_actual_columns_predicted_N_then_Y": matrix.tolist()}
    (reports / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for fig, filename in [
        (plot_confusion_matrices(y_test, y_pred), "01_confusion_matrices.png"),
        (plot_classification_report(report), "02_classification_report.png"),
    ]:
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)
    tn, fp, fn, tp = (int(n) for n in matrix.ravel())
    rows = "\n".join(
        f"| {name.replace('_', ' ')} | {score:.2%} | {baseline_metrics[name]:.2%} |"
        for name, score in metrics.items()
    )
    findings = f"""# Task 4: model evaluation results

Generated by `src/evaluate.py` using Task 3's saved **{provenance['selected_model']}**
pipeline and the original **{len(y_test)} test applications**. No fitting, model
selection, threshold tuning, or new random split occurs here. This re-examines
Task 3's existing test results; it is not an independent new evaluation sample.

## Metrics

Approved (1) is the positive class. Values below are percentages.

| Metric | Selected model | Majority baseline |
| --- | ---: | ---: |
{rows}

The baseline always predicts {provenance['baseline_class']}, the most common
**training** label. It provides a useful reference for the unequal class sizes.

## Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual / predicted | Not Approved (0) | Approved (1) |
| --- | ---: | ---: |
| Not Approved (0) | {tn} (TN) | {fp} (FP) |
| Approved (1) | {fn} (FN) | {tp} (TP) |

- **Accuracy:** {tn + tp}/{len(y_test)} predictions are correct ({metrics['accuracy']:.2%}).
- **Approved precision:** {tp}/{tp + fp} predicted approvals match an actual Approved label ({metrics['precision_approved']:.2%}).
- **Approved recall:** {tp}/{tp + fn} actual Approved records are found ({metrics['recall_approved']:.2%}).
- **False positives:** {fp} Not Approved records are predicted Approved.
- **False negatives:** {fn} Approved records are predicted Not Approved.
- **Not Approved recall:** {tn}/{tn + fp} = {report['Not Approved']['recall']:.2%}.

![Count and row-normalized confusion matrices](figures/01_confusion_matrices.png)

The normalized chart divides each row by that actual class's total. Its diagonal
shows recall for each class. It makes the difference between the two recalls
visible even though there are fewer Not Approved records.

## Classification report

```text
{text_report.rstrip()}
```

![Per-class precision, recall and F1](figures/02_classification_report.png)

**Support** is the number of actual records in each class. **Macro average** gives
each class equal weight. **Weighted average** weights each class by its support,
so the larger class influences it more. **F1** is the harmonic mean of precision
and recall. Undefined ratios are reported as zero (`zero_division=0`); this does
not provide evidence of precision for a class with no predicted examples.

## Interpretation and limits

The model beats the majority baseline on accuracy and balanced accuracy on this
test set. It finds most Approved records, but misses {fp} of {tn + fp} Not Approved
records ({fp / (tn + fp):.2%}). Therefore the high Approved recall does not mean
equally strong performance on both classes. Which error matters more depends on
the purpose of the classifier; that tradeoff has not been specified here.

These labels describe historical approval decisions, not whether loans were
repaid. The source, license, sampling method and dates are unverified; see
[the data record](../data/README.md). This is one small test split, and future
performance may differ. Overall metrics do not measure fairness across groups.
Do not tune against these results: compare changes using training-only validation
and reserve new untouched data for a future final evaluation.

## Reproducibility

The evaluation checks the raw-data fingerprint, dependency versions, unique IDs,
saved train/test assignments, and agreement with Task 3's original predictions.
Test and training IDs are disjoint. `reports/metrics.json` records the data,
model and split-manifest SHA-256 fingerprints. Prediction tables contain only
IDs, labels, scores and correctness; inspect errors in
`reports/misclassified_applications.csv`.
"""
    (reports / "findings.md").write_text(findings, encoding="utf-8")
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\n" + text_report)
    print(f"Verified Task 3 predictions. Saved evaluation to {reports}")


if __name__ == "__main__":
    main()
