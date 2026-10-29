# Backend

Flask app behind the Model Evaluator. For installation, usage and the HTTP API,
see the [main README](../README.md) and the [user guide](../docs/USAGE.md). This
page is for people working on the code.

## Run it

```bash
cd backend
python -m venv venv
# Windows (PowerShell): venv\Scripts\Activate.ps1   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The app runs with Flask's debug reloader on.

## Layout

```
backend/
  app.py                  Flask routes, request validation, global error handlers (BL-07)
  config.py               size limits, allowed extensions, storage paths
  services/
    storage.py            tiny JSON-file index (stand-in for a real database)
    model_service.py      upload_model(), validate_model(), ValidationError
    dataset_service.py    upload_dataset(), validate_dataset()
    evaluation_service.py evaluate_model(), list_evaluations()
  templates/index.html    the page
  static/app.js, style.css
  tests/                  pytest suite
  storage/                uploads and JSON indexes (gitignored, created at runtime)
```

Routes only parse the request and call a service. Services raise
`ValidationError` with a user-facing message, and the routes turn it into a JSON
error with a 4xx status. Anything unexpected is caught by the global handler and
returned as a generic 500, with the details written to the log.

## Configuration

Everything adjustable lives in `config.py`:

| Setting | Default |
|---|---|
| `MAX_MODEL_SIZE_BYTES` | 100 MB |
| `MAX_DATASET_ROWS` | 50,000 |
| `ALLOWED_MODEL_EXTENSIONS` | `.pkl`, `.joblib` |
| `ALLOWED_DATASET_EXTENSIONS` | `.csv` |

`app.py` also caps the request body at 110 MB (the model limit plus headroom for
the form data).

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

The suite runs in a couple of seconds. Every test uses its own temporary
storage folder, so it never touches real uploads in `storage/`.

Test names carry the case IDs from `documents/Web_Based_ML_Model_Evaluator_Test_Cases.pdf`
(`UT` = unit, `IT` = integration, `ST` = system):

| File | Covers | Test plan cases |
|---|---|---|
| `test_model_service.py` | model upload and validation | 04_UT_01 to 04_UT_03 |
| `test_dataset_service.py` | dataset upload and validation | 03_UT_01 to 03_UT_03 |
| `test_evaluation_service.py` | task selection, compatibility, metrics, stored results | 05_*, 06_UT_03, 06_IT_03, 04_IT_02, 04_ST_02 |
| `test_api_workflow.py` | upload, validate, evaluate over HTTP, plus error handling | 03_ST_*, 04_ST_01, 04_IT_03, 05_ST_02, 06_ST_01, 06_ST_02 |
| `test_app.py` | page and static-file smoke tests | n/a |

Metric tests use models that always predict the same value, so the expected
numbers are worked out by hand instead of copied from the code's own output.

## Not implemented yet

Login and roles (FR-01), and the other items in the SRS that are outside the
current sprints.
