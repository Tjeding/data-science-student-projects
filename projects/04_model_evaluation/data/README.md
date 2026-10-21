# Data provenance

Task 4 reuses local artifacts from `projects/03_loan_eligibility/`:

- `data/raw/loan_prediction.csv`: 614 applications, 11 predictors, an ID and target.
- `models/loan_eligibility_pipeline.joblib`: preprocessing and logistic regression
  fitted on the 491 training rows only.
- `reports/split_manifest.csv`: 491 training / 123 test application IDs, created
  with an 80/20 stratified split and random seed 42.
- `reports/metrics.json`: feature names, mapping, selection method, versions and
  raw-data SHA-256 fingerprint.
- `reports/test_predictions.csv`: original held-out predictions checked against
  the loaded pipeline, aligned by application ID.

The target is `Loan_Status`: N = Not Approved (0), Y = Approved (1). Approved is
the positive class. Test support is 38 Not Approved and 85 Approved applications.
The baseline class is determined from training labels only.

The evaluation checks the raw-data fingerprint, unique IDs, saved assignments,
and agreement with Task 3 predictions. No data is downloaded or copied into
Task 4's data folders. Data, model and split-manifest fingerprints are saved in
Task 4's `reports/metrics.json`.

Original source URL, publisher, license, retrieval date, sampling method and
measurement units are **not supplied / unverified**. See
[Task 3's provenance record](../../03_loan_eligibility/data/README.md). Inspection
date: 2026-10-21 (Africa/Johannesburg), not a retrieval date. No redistribution
permission has been established. The raw CSV stays local and unchanged.

Labels describe historical approval decisions rather than observed repayments.
The test set has already been evaluated in Task 3. Task 4 examines those same
results in more detail; it is not a fresh independent evaluation sample.
