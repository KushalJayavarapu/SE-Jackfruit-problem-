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
flask --app app create-admin        # once, to make the first Admin
python app.py
```

Open http://127.0.0.1:5000. The app runs with Flask's debug reloader on.

## Layout

```
backend/
  app.py                   routes, login sessions, CSRF check, error handlers
  config.py                limits, file types, paths, session length
  services/
    db.py                  SQLite schema and small query helpers
    auth_service.py        register, log in (FR-01)
    admin_service.py       suspend, reactivate, roles, audit log (FR-08)
    model_service.py       upload and validate models (BL-01, BL-02)
    dataset_service.py     upload and validate datasets (BL-03, BL-04)
    compatibility.py       does a model's input match a dataset's columns?
    metrics.py             metric definitions per task type
    evaluation_service.py  run an evaluation, reopen, compare (BL-05, FR-06)
    visualization_service.py  chart data and PNG rendering (FR-05)
    report_service.py      PDF reports (FR-07)
    errors.py, timeutil.py
  templates/               Jinja pages (base, login, dashboard, ...)
  static/app.js, style.css
  sample_data/             sample models and datasets
  tests/                   pytest suite
  storage/                 evaluator.db, uploaded files, .secret_key (gitignored)
```

Routes only read the request and call a service. Services raise
`ValidationError` (or `NotFoundError`, `ForbiddenError`) with a message meant for
the user, and the routes turn it into a JSON error with a 4xx status. Anything
unexpected is caught by the global handler, logged, and returned as a generic 500.

## How access control works

- Every route except login, registration and static files needs a signed-in user.
- The user is read from the database on every request, so suspending an account
  or changing its role takes effect immediately.
- Every model, dataset and evaluation has an `owner_id`. Service functions take
  the owner and include it in every lookup, so another user's record is
  indistinguishable from one that doesn't exist (404).
- Admin routes use the `admin_required` decorator, and the admin service checks
  the actor's role again.
- Changing requests (POST, DELETE) need the CSRF token, sent as the
  `X-CSRF-Token` header or a `csrf_token` form field.

## Extending it (NFR-05)

- **New metric:** add it to the task's function in `services/metrics.py` and to `METRIC_INFO`.
- **New chart:** add its data in `visualization_service.build_chart_data`, a
  drawer in `DRAWERS` and a title in `CHART_TITLES`.
- **New task type:** add a function to `metrics.TASKS` and a rule in `evaluation_service._check_task_compatibility`.

None of these touch upload, storage or login code.

## Configuration

Settings are in `config.py`. Environment variables:

| Variable | Purpose |
|---|---|
| `EVALUATOR_SECRET_KEY` | Session signing key. A random one is saved in `storage/.secret_key` if unset. |
| `EVALUATOR_DATABASE` | Path of the SQLite file. Defaults to `storage/evaluator.db`. |

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests
```

Each test gets its own temporary database and upload folders. Test names carry
the case IDs from `documents/Web_Based_ML_Model_Evaluator_Test_Cases.pdf`
(`UT` = unit, `IT` = integration, `ST` = system).

| File | Covers |
|---|---|
| `test_auth.py` | registration, login, suspended accounts, sessions, redirects, CSRF (01_*, 02_*) |
| `test_admin.py` | account and role rules, audit log, first-admin command (12_*, 13_*) |
| `test_isolation.py` | one Analyst can't reach another's data (NFR-02) |
| `test_model_service.py`, `test_dataset_service.py` | upload and validation (03_*, 04_*) |
| `test_compatibility_at_upload.py` | model/dataset fit reported at validation time |
| `test_evaluation_service.py` | task selection, compatibility, metrics, stored results (05_*, 06_*) |
| `test_charts_compare_reports.py` | charts, reopening, comparison, PDF reports (07_* to 11_*) |
| `test_api_workflow.py` | full workflow over HTTP, request validation, error handling |
| `test_performance.py` | 50,000-row evaluations finish in under 10 seconds (NFR-01) |
| `test_app.py` | pages render |

Metric tests use models that always predict the same value, so expected numbers
are worked out by hand instead of copied from the code's own output.
