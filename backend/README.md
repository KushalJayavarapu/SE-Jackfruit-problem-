# Backend — Sprint 1 (BL-01, BL-02, BL-03)

Flask app implementing the first 3 backlog items:

- **BL-01** — upload a model file (`.pkl` / `.joblib`, ≤100 MB)
- **BL-02** — validate an uploaded model (loads it, confirms it exposes
  `predict()`, reports the inputs it expects)
- **BL-03** — upload a test dataset (`.csv`, ≤50,000 rows) with a target
  variable

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
    dataset_service.py      # BL-03 upload_dataset()
  templates/index.html
  static/style.css, app.js
  storage/                # uploaded files + JSON indexes land here (gitignored)
```

## Not yet in scope

Login/roles (FR-01), model↔dataset compatibility checking (back half of
FR-02), running evaluations and metrics (BL-04/BL-05) — later sprint items.
