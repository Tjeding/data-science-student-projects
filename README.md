# Data Science Student Projects

A growing portfolio of independent Python data science projects. Titanic EDA and housing regression have starting scripts, and the loan eligibility project includes a guided notebook, reproducible model comparison, and a findings report. Other projects currently provide scaffolds.

## Project index

| ID | Project | Folder |
| --- | --- | --- |
| 01 | Exploratory Data Analysis on a Public Dataset | [01_public_dataset_eda](projects/01_public_dataset_eda/README.md) |
| 02 | Linear Regression on Housing Prices | [02_housing_linear_regression](projects/02_housing_linear_regression/README.md) |
| 03 | Loan Eligibility Prediction | [03_loan_eligibility](projects/03_loan_eligibility/README.md) |
| 04 | Model Evaluation | [04_model_evaluation](projects/04_model_evaluation/README.md) |
| 05 | Iris Data Exploration | [05_iris_data_exploration](projects/05_iris_data_exploration/README.md) |
| 06 | Analyzing and Visualizing Movie Ratings | [06_movie_ratings](projects/06_movie_ratings/README.md) |
| 07 | Titanic Survival Portfolio Project | [07_titanic_survival](projects/07_titanic_survival/README.md) |
| 08 | COVID-19 Data Analysis | [08_covid19_analysis](projects/08_covid19_analysis/README.md) |
| 09 | YouTube Data Dashboard with Streamlit | [09_youtube_dashboard](projects/09_youtube_dashboard/README.md) |

The nine projects use unique IDs in the order supplied, even though the original list restarts numbering for each group.

## Repository layout

```text
projects/             Nine independent project folders
  <id>_<name>/
    README.md         Scope, status, and planned work
    data/             Provenance and raw/interim/processed data
    notebooks/        Numbered exploration and modeling notebooks
    src/              Python code for this project
    models/           Generated model artifacts
    reports/figures/  Findings, tables, and charts
    app/              Dashboard projects only
shared/               Utilities shared by multiple projects when needed
templates/project/   Copyable scaffold for future projects
requirements.txt      Common analysis and notebook dependencies
requirements-dashboard.txt  Optional dashboard dependencies
```

## Environment setup

From the repository root in PowerShell, using a Python version supported by the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

For future dashboard work, also install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
```

Dependency files are starting lists, not a tested lockfile. Record exact versions after validating each project's environment. Add dependencies unique to a project in its own `requirements.txt` and install them into that project's environment if needed.

## Existing work

- `Task1.py` moved to `projects/01_public_dataset_eda/src/eda.py`.
- `titanic.csv` and `titanic_cleaned.csv` moved to that project's `data/raw/` and `data/processed/`.
- `Task2.py` moved to `projects/02_housing_linear_regression/src/train.py`.
- `BostonHousing.csv` moved to that project's `data/raw/`.

The scripts resolve data and output paths relative to their own project folder. After installing dependencies, run them from the repository root:

```powershell
.\.venv\Scripts\python.exe projects/01_public_dataset_eda/src/eda.py
.\.venv\Scripts\python.exe projects/02_housing_linear_regression/src/train.py
.\.venv\Scripts\python.exe projects/03_loan_eligibility/src/train.py
```

These are learning implementations. Titanic missing-value handling supports current pandas behavior, and housing preprocessing is fitted on the training split only. The housing script uses multiple predictors and reports MAE, MSE, RMSE, and R-squared. The Titanic EDA script opens plot windows. The loan script saves plots and compares a majority baseline, logistic regression, a decision tree, and a random forest using training-only cross-validation. Its [guided notebook](projects/03_loan_eligibility/notebooks/01_loan_eligibility.ipynb) explains each step. Generated CSV outputs go into `data/processed/` or `reports/`.

## Add a project

Copy the template into a new, uniquely numbered folder:

```powershell
Copy-Item -Recurse templates/project projects/10_your_project_name
```

Update its README and data provenance record, then add it to the project index. Add an `app/` folder for dashboards and a `tests/` folder when there is reusable behavior to test. Move code into `shared/` only when multiple projects need it.

## Working conventions

- Keep original data unchanged and document its source, license, and retrieval date.
- New datasets, generated outputs, environments, and secrets are ignored by default. Existing source CSVs are explicitly retained.
- Clear large notebook outputs before committing. Use names such as `01_exploration.ipynb` and `02_modeling.ipynb`.
- Resolve paths explicitly; scripts should use `Path(__file__)` and notebooks should document their working directory.
- Split training and test data before fitting preprocessing, and record random seeds and evaluation choices.
- Store API credentials in environment variables or local ignored configuration files.
