# Backend — BL-01, BL-02

- **BL-01** — upload a model file (`.pkl` / `.joblib`, ≤100 MB)
- **BL-02** — validate an uploaded model: loads it, confirms it exposes
  `predict()`, reports the inputs it expects (`n_features_in_`,
  `feature_names_in_` where available)

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
    model_service.py    # BL-01 upload_model(), BL-02 validate_model()
  templates/index.html
  static/style.css, app.js
  storage/               # uploaded files + JSON index land here (gitignored)
```

## Not yet in scope

Test dataset upload (BL-03), login/roles (FR-01), model↔dataset
compatibility checking, running evaluations — later sprint items.
