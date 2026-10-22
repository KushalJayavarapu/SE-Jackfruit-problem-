# Backend — BL-01: Upload a machine learning model

Flask app, first backlog item only.

- **BL-01** — upload a model file (`.pkl` / `.joblib`, ≤100 MB), with
  per-file format and size checks (FR-02, NFR-01).

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
  app.py
  config.py
  services/
    storage.py          # tiny JSON-file index (placeholder for a real DB)
    model_service.py    # BL-01: upload_model()
  templates/index.html
  static/style.css, app.js
  storage/               # uploaded files + JSON index land here (gitignored)
```
