# Web-Based ML Model Evaluator

A small web app for evaluating a trained machine-learning model against a test
dataset. Upload a model and a CSV dataset, validate both, pick a task type
(classification or regression), and get the metrics back in the browser. Every
run is saved to an evaluation history.

Built for the Software Engineering course project (team SE-Jackfruit).

## What it does

| Step | What you do | What the app checks |
|---|---|---|
| 1. Model | Upload a `.pkl` or `.joblib` file (up to 100 MB), then validate it | File type and size, that it loads, that it has a `predict()` method |
| 2. Dataset | Upload a `.csv` (up to 50,000 rows) and name the target column | File type and row limit, that the target column exists, no missing target values, at least one feature column |
| 3. Evaluate | Pick a validated model, a validated dataset and a task type | Model and dataset columns match, the task type fits the model and target |
| 4. Results | Read the metrics and the history table | - |

Metrics reported:

- **Classification:** accuracy, precision, recall, F1 (all weighted) and ROC-AUC
- **Regression:** MAE, MSE, RMSE and R²

## System requirements

- Python 3.10 or newer (developed and tested on 3.13)
- pip
- A modern browser (Chrome, Edge, Firefox or Safari)
- About 500 MB of free disk space for the Python packages
- Windows, macOS or Linux

No database or other service is needed. Uploads and results are stored as files
inside `backend/storage/`.

## Quick start

```bash
git clone https://github.com/KushalJayavarapu/SE-Jackfruit-problem-.git
cd SE-Jackfruit-problem-/backend

python -m venv venv
# Windows (PowerShell):  venv\Scripts\Activate.ps1
# Windows (cmd):         venv\Scripts\activate.bat
# macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser. Press `Ctrl+C` in the terminal to stop.

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope Process Bypass` once in that window and try again.

## Using the app

The full walkthrough is in **[docs/USAGE.md](docs/USAGE.md)**. It covers how to
create a model and dataset the app accepts, what each metric means, every error
message and its fix, and the HTTP API.

In short: upload and validate a model, upload and validate a dataset, choose a
task type, press **Run evaluation**.

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

Tests use temporary storage and never touch your real uploads.

## Project layout

```
backend/
  app.py              Flask routes and global error handling
  config.py           size limits, allowed file types, storage paths
  services/           model, dataset and evaluation logic, JSON-file storage
  templates/, static/ the web page (HTML, CSS, JavaScript)
  tests/              pytest suite
  storage/            uploaded files and results (created at runtime, not in git)
documents/            SRS, diagrams, test plan, backlog
docs/USAGE.md         user guide and API reference
```

## Security note

Model files are Python pickles. **Loading a pickle can run arbitrary code**, so
only upload model files you created yourself or fully trust. The app is meant
for local use by its owner; it has no login yet, so do not expose it to the
internet.

## Known limitations

- No login or user roles yet (FR-01 in the SRS is not implemented).
- Only scikit-learn style models are supported: the model must have `predict()`,
  and `predict_proba()` for ROC-AUC.
- Only CSV datasets, and only the two task types above.
- Data is kept in local JSON files, not a database.
