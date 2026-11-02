# Web-Based ML Model Evaluator

A web app for evaluating trained machine-learning models. Analysts register,
upload a model and a test dataset, pick a task type (classification or
regression) and get metrics and charts. Every run is saved, so evaluations can
be reopened, compared side by side and exported as a PDF report. Admins manage
accounts and roles, with an audit log of every change.

Built for the Software Engineering course project (team SE-Jackfruit).

## What it does

| Area | What you can do |
|---|---|
| Accounts | Register as an Analyst, log in, log out. Admins create other Admins, suspend or reactivate accounts and change roles. |
| Models | Upload `.pkl` or `.joblib` files (up to 100 MB), validate them, see which columns they expect and which of your datasets they fit. |
| Datasets | Upload a `.csv` (up to 50,000 rows), pick the target column from a list, validate it. |
| Evaluation | Pick a model, a dataset and a task type. The app checks they fit together first, then runs the model. |
| Results | Metrics, a confusion matrix and ROC curve (classification) or residual and predicted-vs-actual plots (regression). |
| History | Every evaluation is saved. Reopen it later with the same numbers and charts. |
| Compare | Tick two or more evaluations of the same task type and see their metrics side by side. |
| Reports | Download a PDF for one evaluation or several, including the charts. |
| Privacy | Each Analyst sees only their own models, datasets and evaluations. |

Metrics: accuracy, precision, recall, F1 (weighted) and ROC-AUC for
classification; MAE, MSE, RMSE and R² for regression.

## System requirements

- Python 3.10 or newer (developed and tested on 3.13)
- pip
- A modern browser (Chrome, Edge, Firefox or Safari)
- About 600 MB of free disk space for the Python packages
- Windows, macOS or Linux

No database server is needed. Accounts, results and the audit log live in a
SQLite file at `backend/storage/evaluator.db`; uploaded files are stored next to it.

## Quick start

```bash
git clone https://github.com/KushalJayavarapu/SE-Jackfruit-problem-.git
cd SE-Jackfruit-problem-/backend

python -m venv venv
# Windows (PowerShell):  venv\Scripts\Activate.ps1
# Windows (cmd):         venv\Scripts\activate.bat
# macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt

# Create the first Admin account (asks for name, email and password)
flask --app app create-admin

python app.py
```

Open http://127.0.0.1:5000. Log in as the Admin, or choose **Create an Analyst
account** to register. Press `Ctrl+C` in the terminal to stop the app.

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope Process Bypass` once in that window and try again.

**Want to try it quickly?** `backend/sample_data/` has two ready-made models and
datasets. See its README.

## Accounts and roles

- **Registration is open but only creates Analysts.** No form or request can
  create an Admin.
- **The first Admin** is created from the command line with `flask --app app create-admin`.
- **More Admins** are made by an existing Admin, either by changing an account's
  role or by creating an account with the Admin role.
- Admins can't suspend or change their own account, and the last active Admin
  can never be suspended or demoted.
- A suspended account is logged out immediately and can't log in until reactivated.
- Sessions last 8 hours.

## More documentation

- **[docs/USAGE.md](docs/USAGE.md)**: user guide with file preparation, the
  workflow, metrics, every error message and its fix, and the HTTP API.
- **[backend/README.md](backend/README.md)**: code layout, settings and tests.
- **[documents/](documents/)**: SRS, use cases, diagrams, security requirements, test plan and backlog.

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

About 220 tests, around 20 seconds. Tests use temporary storage and a temporary
database, and never touch your real data.

## Project layout

```
backend/
  app.py               Flask routes, sessions, CSRF, error handling
  config.py            limits, allowed file types, storage paths
  services/            accounts, admin, models, datasets, evaluation, charts, reports
  templates/, static/  the web pages (HTML, CSS, JavaScript)
  sample_data/         ready-made models and datasets to try
  tests/               pytest suite
  storage/             database and uploaded files (created at runtime, not in git)
docs/USAGE.md          user guide and API reference
documents/             project documents
```

## Security notes

- **Model files are Python pickles. Loading one can run arbitrary code.** Only
  upload models you created or fully trust.
- Passwords are stored hashed. Forms and API calls are protected against CSRF.
- Set `EVALUATOR_SECRET_KEY` to a long random value if you run the app anywhere
  other than your own machine. Otherwise a key is generated and saved in `backend/storage/`.
- The built-in server (`python app.py`) is for development and demos. For a
  real deployment put the app behind a production server and HTTPS.

## Known limitations

- No password reset or email verification (there is no email service).
- Only scikit-learn style models: `predict()` is required, and `predict_proba()` for ROC-AUC.
- Only CSV datasets and the two task types above.
- Evaluations of very large models run inside the web request. The 10-second
  target (NFR-01) was measured with typical models (logistic regression,
  random forest, linear regression) on 50,000 rows; a heavy model near the
  100 MB limit may be slower.
- Evaluation charts for regression show a random sample of 2,000 rows when the
  dataset is larger. Metrics always use every row.
