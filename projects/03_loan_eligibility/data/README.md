# Data provenance

- Local input: `raw/loan_prediction.csv` (38,013 bytes when inspected).
- Availability: already in the user's workspace; not downloaded by this project.
- Source URL, publisher, version, license and retrieval date: **not supplied / unverified**.
- Inspection date: 2026-10-21 (Africa/Johannesburg), not a retrieval date.
- Shape: 614 applications, 13 columns (ID, 11 features and target).
- Target: `Loan_Status`; Y = Approved, N = Not Approved; encoded as 1 and 0.
- Schema: see the feature dictionary in the [project README](../README.md).
- Units: currency, income period, amount scale and term unit are unconfirmed.
- `Credit_History`: observed codes 0 and 1; original definition not supplied.
- Quality: 149 missing feature cells; no missing targets, identical duplicate
  rows or repeated IDs in this copy.
- Limitations: unknown dates, geography, sampling and approval policy; potential
  historical and demographic bias; approval is not repayment.
- Reproduction: retain the exact CSV at the input path. The script records its
  SHA-256 fingerprint in `reports/metrics.json`. An upstream download cannot
  be reproduced until a source is supplied.

Keep the raw file unchanged and local under the existing Git ignore rules.
Redistribution permission has not been established. Preprocessing is fitted
within training data only; generated outputs go to `reports/` and `models/`.
