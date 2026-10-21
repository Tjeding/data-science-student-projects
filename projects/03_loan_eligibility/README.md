# Loan Eligibility Prediction: a guided Python project

Start with the [results report and charts](reports/findings.md), which can be
read on GitHub, or open [the annotated notebook](notebooks/01_loan_eligibility.ipynb)
to run and modify the code. It explains each stage from reading the supplied CSV through exploration,
preprocessing, model comparison, evaluation, and prediction for a fictional application.
The [results report](reports/findings.md) contains actual findings and charts.
The [Python script](src/train.py) reproduces the analysis and saves a fitted model.

## Run the project

From the repository root in PowerShell:

```powershell
# Use the existing environment; create it first if needed: python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r projects/03_loan_eligibility/requirements.txt
.\.venv\Scripts\python.exe projects/03_loan_eligibility/src/train.py
.\.venv\Scripts\python.exe -m jupyterlab projects/03_loan_eligibility/notebooks/01_loan_eligibility.ipynb
```

Use the environment's Python kernel. In Jupyter, choose **Kernel > Restart Kernel
and Run All Cells**. The notebook finds the project from the repository root,
project folder, or notebook folder. The script works from any working directory
and saves plots without opening windows. The raw CSV stays unchanged.

## Understand the problem

One row is one loan application. `Loan_Status` is the **target**, the answer we
want to predict: `Y` means Approved, `N` means Not Approved. Predicting one of two
classes is **binary classification**. Because training examples have known answers,
this is **supervised learning**. Input variables are **features**, written `X`;
answers are written `y`. We encode N as 0 and Y as 1.

The model learns recorded historical approval decisions. These labels do not tell
us whether a loan was repaid and do not define objective creditworthiness.

## Feature dictionary

The supplied file has **614 rows and 13 columns**: 11 predictors, an ID and a target.

| Column | Meaning supported by names / observed values | Role |
| --- | --- | --- |
| `Loan_ID` | Unique application identifier | Excluded |
| `Gender` | Recorded Male / Female category | Categorical |
| `Married` | Yes / No | Categorical |
| `Dependents` | 0, 1, 2, or 3+ | Categorical |
| `Education` | Graduate / Not Graduate | Categorical |
| `Self_Employed` | Yes / No | Categorical |
| `ApplicantIncome` | Applicant income | Numeric |
| `CoapplicantIncome` | Coapplicant income | Numeric |
| `LoanAmount` | Loan amount | Numeric |
| `Loan_Amount_Term` | Loan term | Numeric |
| `Credit_History` | Binary code 0 / 1 | Binary |
| `Property_Area` | Rural / Semiurban / Urban | Categorical |
| `Loan_Status` | Y = Approved, N = Not Approved | Target |

Currency, income period, amount scale, term unit and the exact credit-history
definition are unconfirmed. Do not calculate income-to-loan ratios until units
are established. No source or license accompanied the CSV; see
[the provenance record](data/README.md).

## Explore before changing the data

`pd.read_csv()` creates a pandas **DataFrame**, a table in Python. `head()` previews
rows, `shape` gives row and column counts, `info()` shows types and nonmissing
counts, and `describe()` summarizes distributions. Text columns can appear as
`str`, `string`, or `object`, depending on the pandas version.

There are **422 Approved (68.73%)** and **192 Not Approved (31.27%)** applications.
A model that always predicts Approved therefore gets about 69% accuracy while
detecting no Not Approved applications. That is why we need a majority baseline
and metrics beyond accuracy.

| Feature | Missing cells |
| --- | ---: |
| `Gender` | 13 |
| `Married` | 3 |
| `Dependents` | 15 |
| `Self_Employed` | 32 |
| `LoanAmount` | 22 |
| `Loan_Amount_Term` | 14 |
| `Credit_History` | 50 |

There are **149 missing cells**, no missing targets, no identical duplicate rows,
and no repeated IDs. One row can contain several missing cells. Zero coapplicant
income is a recorded value, not automatically a missing value. We fill feature
gaps during training and never guess missing target labels.

Income and loan amount have long right tails. Applicant income has a median of
3,812.5 and maximum of 81,000 in the supplied units. Unusual values are not
automatically errors; deleting them requires evidence.

For `Credit_History=1`, 378 of 475 applications (79.58%) are Approved; for code 0,
7 of 89 (7.87%) are Approved. Another 50 have missing credit history, of which 37
are Approved. This is association in these records, not causation.

Full-data plots describe the supplied file. They are not used to tune models or
select features. For a strict prospective study, reserve the test data before
target-driven exploration and examine relationships on training data alone.

## Split and preprocess without leakage

We use an **80/20 stratified split**: 491 training rows and 123 test rows.
Stratification preserves approximate class proportions; `random_state=42` makes
the split repeatable for the same data and software. Keep the test data aside
while comparing models. Exclude `Loan_ID` and `Loan_Status` from the features.

**Data leakage** means training learns information that should be unavailable.
Even computing a median from the entire dataset leaks information about test
rows. A scikit-learn `Pipeline` keeps preprocessing inside training, including
each cross-validation fold, following the
[official leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

A `ColumnTransformer` applies separate rules:

1. Numeric fields: fill missing values with the training median, then subtract
   the training mean and divide by its standard deviation (`StandardScaler`).
   This helps regularized logistic regression handle different units. Scaling
   does not remove skew or outliers; the median resists extremes better than the mean.
2. Categories: fill with the most frequent training category, then one-hot encode
   into 0/1 columns. Assigning Urban, Rural, Semiurban to 0, 1, 2 would imply
   unsupported numeric distances. Keep `3+` dependents as a category.
3. Binary credit history: fill with its most frequent training code and keep 0/1.
   This is a simple assumption, not evidence that missing history means the mode.

`handle_unknown="ignore"` allows new categories at prediction time by encoding
them as zeros for that field. The model has not learned what those values mean.
Trees do not require scaling, but we use one preprocessing design for all models
to simplify comparison. We never manually encode new applications differently.

## Compare models and evaluate

| Model | How it works | Learning point |
| --- | --- | --- |
| Majority baseline | Always predicts the common training class | Minimum reference |
| Logistic regression | Weighted features passed through a logistic function | A classifier despite its name |
| Decision tree | A sequence of feature-based splits | Limit depth to reduce overfitting |
| Random forest | Averages many randomized trees | Complexity does not guarantee improvement |

We compare fixed example settings using **five-fold stratified cross-validation**
on training data: train on four folds, validate on the fifth, repeat five times.
Preprocessing is learned afresh inside each training fold. We choose the model
with the highest **mean balanced accuracy**, decided before examining test scores.
Fold standard deviation indicates variability; it is not a confidence interval.

Fit the chosen model on all 491 training rows, then evaluate on the 123 held-out
rows. We also evaluate the majority baseline for reference. We do not choose a
model or threshold by repeatedly trying alternatives on the test set.

**Approved (1) is the positive class**. A confusion matrix uses actual labels in
rows and predicted labels in columns. True negatives are correct Not Approved
predictions; false positives predict Approved for a Not Approved label; false
negatives predict Not Approved for an Approved label; true positives are correct
Approved predictions.

| Metric | Interpretation |
| --- | --- |
| Accuracy | Correct predictions / all predictions |
| Approved precision | Actually Approved / all predicted Approved: TP / (TP + FP) |
| Approved recall | Predicted Approved / all actually Approved: TP / (TP + FN) |
| Approved F1 | Harmonic mean of approved precision and recall |
| Balanced accuracy | Average recall across Approved and Not Approved |
| ROC AUC | How well scores rank Approved above Not Approved across thresholds |

Read both classes in the classification report. High Approved recall alone can
conceal weak Not Approved recall. ROC AUC is not accuracy. A model score of 0.8
does not automatically establish an 80% real-world chance; calibration and
representative future data would be needed. See [actual results](reports/findings.md).

## Saved model and outputs

The notebook predicts a fictional one-row application using the same raw columns.
`predict()` returns a class; `predict_proba()` returns class scores. Both reuse the
fitted preprocessing. The script saves preprocessing and the classifier together
in `models/loan_eligibility_pipeline.joblib`, fitted on training rows only.

Load only trusted joblib files, because deserialization can execute code. Use
matching library versions, recorded with every run in `reports/metrics.json`.

- `reports/findings.md`: findings, confusion matrix and charts.
- `reports/data_overview.csv`, `descriptive_statistics.csv`, `target_counts.csv`,
  `credit_history_counts.csv`: exploration tables.
- `reports/cv_results.csv`: training-only model comparison.
- `reports/metrics.json`: scores, seed, feature list, dataset hash and versions.
- `reports/classification_report.txt`, `test_predictions.csv`: evaluation details.
- `reports/split_manifest.csv`: application IDs in each partition.
- `reports/figures/`: four saved figures.
- `models/loan_eligibility_pipeline.joblib`: fitted model and preprocessing.

PNG charts and the Markdown findings report are included in version control for
GitHub. Generated tables, model files and the local CSV remain ignored. The
notebook has cleared outputs; run it locally to see its output, or read the
findings report for the saved charts. The executed notebook and HTML lesson
are local exports. Generate them and open the HTML in your browser with:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute projects/03_loan_eligibility/notebooks/01_loan_eligibility.ipynb --output loan_eligibility_executed.ipynb --output-dir projects/03_loan_eligibility/reports --ExecutePreprocessor.timeout=180
.\.venv\Scripts\python.exe -m jupyter nbconvert --to html projects/03_loan_eligibility/reports/loan_eligibility_executed.ipynb --output loan_eligibility_walkthrough.html --output-dir projects/03_loan_eligibility/reports --embed-images
```

## Practice and limitations

1. Explain why 69% accuracy can be unhelpful here.
2. Find the feature with the most missing values and explain how we fill it.
3. Describe each confusion-matrix cell in words.
4. Explain why the imputer must be fitted again in every cross-validation fold.
5. As a new training-only experiment, compare a log transformation of income or
   a missingness indicator using CV. Repeated revisions after viewing test results
   require a fresh final evaluation set.

This small dataset has unknown sampling, dates and source. Gender, marital status
and other fields can reproduce historical bias; dropping one field alone cannot
establish fairness. These metrics do not establish repayment risk, fairness, or
suitability for real lending. Test performance from one split is uncertain and
can change on future applicants.

References: [pandas CSV loading](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html),
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[cross-validation API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).
