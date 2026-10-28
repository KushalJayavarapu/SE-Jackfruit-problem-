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

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

The suite has about 90 tests and runs in a couple of seconds. Every test uses its own
temporary storage folder, so it never touches real uploads in `storage/`.

Test names carry the case IDs from `documents/Web_Based_ML_Model_Evaluator_Test_Cases.pdf`
(`UT` = unit, `IT` = integration, `ST` = system), so each test can be traced back to the plan:

| File | Covers | Test plan cases |
|---|---|---|
| `test_model_service.py` | model upload and validation (BL-01, BL-02) | 04_UT_01 to 04_UT_03 |
| `test_dataset_service.py` | dataset upload and validation (BL-03, BL-04) | 03_UT_01 to 03_UT_03 |
| `test_evaluation_service.py` | task selection, compatibility, metrics, stored results (BL-05) | 05_*, 06_UT_03, 06_IT_03, 04_IT_02, 04_ST_02 |
| `test_api_workflow.py` | full upload -> validate -> evaluate flow over HTTP, request validation and error handling (BL-07) | 03_ST_*, 04_ST_01, 04_IT_03, 05_ST_02, 06_ST_01, 06_ST_02 |
| `test_app.py` | page and static-file smoke tests | n/a |

Metric tests use models that always predict the same value, so the expected numbers
are worked out by hand instead of being copied from the code's own output.
