# Task 4: Model Evaluation

Start with the [executed lesson](reports/model_evaluation_walkthrough.html), or
open the [guided notebook](notebooks/01_model_evaluation.ipynb) to run the code.
The [findings report](reports/findings.md) explains the actual results.

## Status

Complete: reproducible evaluation, a guided notebook, count and normalized
confusion matrices, classification reports, and a majority-baseline comparison.

## Data

Reuses Task 3's saved logistic regression pipeline and original 123 test
applications. The pipeline was fitted on 491 training applications. The saved
split manifest restores the partition by application ID. Task 4 checks the raw
data fingerprint and reproduces Task 3's predictions without fitting or tuning.
These are the same test results already evaluated in Task 3, examined in detail.
See [data provenance](data/README.md).

## Completed work

- [x] Define Approved (1) as positive and Not Approved (0) as negative.
- [x] Calculate accuracy, precision, recall, F1 and balanced accuracy.
- [x] Visualize count and row-normalized confusion matrices.
- [x] Produce text, CSV and heatmap classification reports.
- [x] Discuss class imbalance, error types, baseline results and limitations.

## Run the project

From the repository root in PowerShell, use the same environment as Task 3:

```powershell
.\.venv\Scripts\python.exe -m pip install -r projects/04_model_evaluation/requirements.txt
.\.venv\Scripts\python.exe projects/04_model_evaluation/src/evaluate.py
.\.venv\Scripts\python.exe -m jupyterlab projects/04_model_evaluation/notebooks/01_model_evaluation.ipynb
```

In Jupyter, select the environment's Python kernel, then choose **Kernel > Restart
Kernel and Run All Cells**. The notebook finds the project from inside the
repository. The script works from any working directory and saves plots without
opening windows.

Task 3's raw CSV, saved pipeline, metrics, split manifest and predictions already
exist locally. If generated artifacts are missing, restore the original CSV and
run `projects/03_loan_eligibility/src/train.py` first. Task 4 checks the recorded
dependency versions before loading the locally created joblib file. Only load
trusted model files.

## Understand the results

| Metric | Selected model | Majority baseline |
| --- | ---: | ---: |
| Accuracy | 86.18% | 69.11% |
| Approved precision | 84.00% | 69.11% |
| Approved recall | 98.82% | 100.00% |
| Approved F1 | 90.81% | 81.73% |
| Balanced accuracy | 78.36% | 50.00% |

The model correctly classifies **106 of 123** test applications. The baseline
always predicts Approved, the most frequent label in the training set.

Rows in the confusion matrix are actual labels; columns are predictions:

| Actual / predicted | Not Approved (0) | Approved (1) |
| --- | ---: | ---: |
| Not Approved (0) | 22 true negatives | 16 false positives |
| Approved (1) | 1 false negative | 84 true positives |

Approved recall is high, but Not Approved recall is only **22/38 = 57.89%**.
This uneven performance is why accuracy alone is insufficient. In the report,
**support** is the actual number of examples in each class. **Macro** averages
give each class equal weight; **weighted** averages use class sizes.

The labels describe historical approvals, not repayment outcomes. This single
small test split does not establish future performance or fairness. The data
source, license, dates and sampling are unverified.

## Saved outputs

The script saves `reports/findings.md`, `metrics.json`, `metrics_comparison.csv`,
`confusion_matrix.csv`, `classification_report.txt`, `classification_report.csv`,
`test_predictions.csv`, and `misclassified_applications.csv`. Input fingerprints
and versions are recorded in `metrics.json`. Overall accuracy appears in the
text report and metrics files; the report CSV contains class rows and averages.
The two PNG charts are in `reports/figures/`.

Generated artifacts are ignored by Git under the existing repository rules;
the source notebook keeps its outputs clear. The executed notebook and HTML
lesson were generated during validation. To regenerate them:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=python3 --output model_evaluation_executed --output-dir projects/04_model_evaluation/reports projects/04_model_evaluation/notebooks/01_model_evaluation.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html --output model_evaluation_walkthrough --output-dir projects/04_model_evaluation/reports projects/04_model_evaluation/reports/model_evaluation_executed.ipynb
```

Use the environment's Python kernel. The HTML embeds the notebook charts so
the lesson can be read without running Python.

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
