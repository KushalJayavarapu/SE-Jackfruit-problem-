# Backend — BL-01 – BL-05

Flask app implementing the first 5 backlog items:

- **BL-01** — upload a model file (`.pkl` / `.joblib`, ≤100 MB)
- **BL-02** — validate an uploaded model (loads it, confirms it exposes
  `predict()`, reports the inputs it expects)
- **BL-03** — upload a test dataset (`.csv`, ≤50,000 rows) with a target
  variable
- **BL-04** — validate an uploaded dataset (re-reads it, reports
  per-column dtype/missing-value counts, and flags it invalid if the
  target column has missing values or there are no feature columns)
- **BL-05** — evaluate a validated model on a validated dataset for a
  chosen task type. Checks model/dataset/task compatibility first, then
  computes classification (accuracy, precision, recall, F1, ROC-AUC) or
  regression (MAE, MSE, RMSE, R²) metrics and stores the result.

Implements the per-file format checks and size limits from FR-02 and
NFR-01 in the project's SRS.

## Run it

```bash
cd backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Layout

```
backend/
  app.py                  # Flask routes
  config.py                # limits, file paths
  services/
    storage.py             # tiny JSON-file index (placeholder for a real DB)
    model_service.py        # BL-01 upload_model(), BL-02 validate_model()
    dataset_service.py      # BL-03 upload_dataset(), BL-04 validate_dataset()
    evaluation_service.py   # BL-05 evaluate_model()
  templates/index.html
  static/style.css, app.js
  storage/                # uploaded files + JSON indexes land here (gitignored)
```

## Not yet in scope

Login/roles (FR-01), displaying/visualizing metrics (BL-06), friendlier error handling (BL-07),
UI polish (BL-08) — later items.
