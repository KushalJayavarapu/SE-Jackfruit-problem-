"""
Central settings. Services read these as `config.NAME` at call time, so tests
can point them at a temporary folder with monkeypatch.
"""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

MODELS_DIR = os.path.join(STORAGE_DIR, "models")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")

# SQLite database holding users, uploads, evaluations and the audit log.
DATABASE_PATH = os.environ.get("EVALUATOR_DATABASE", os.path.join(STORAGE_DIR, "evaluator.db"))

# NFR-01: model file up to 100 MB, dataset up to 50,000 rows
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024
MAX_DATASET_ROWS = 50_000

ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib"}
ALLOWED_DATASET_EXTENSIONS = {".csv"}

# Accounts (FR-01)
MIN_PASSWORD_LENGTH = 8
SESSION_HOURS = 8

# Charts: how many points are kept per evaluation for plotting (FR-05)
MAX_RESIDUAL_POINTS = 2000
MAX_ROC_POINTS = 400
MAX_ROC_CLASSES = 10

for d in (MODELS_DIR, DATASETS_DIR):
    os.makedirs(d, exist_ok=True)


def secret_key():
    """Use EVALUATOR_SECRET_KEY if set, otherwise a random key kept in storage/."""
    env_key = os.environ.get("EVALUATOR_SECRET_KEY")
    if env_key:
        return env_key
    key_path = os.path.join(STORAGE_DIR, ".secret_key")
    if not os.path.exists(key_path):
        with open(key_path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(key_path) as f:
        return f.read().strip()
