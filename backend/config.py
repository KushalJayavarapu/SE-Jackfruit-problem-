import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

MODELS_DIR = os.path.join(STORAGE_DIR, "models")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")

MODELS_INDEX = os.path.join(STORAGE_DIR, "models.json")
DATASETS_INDEX = os.path.join(STORAGE_DIR, "datasets.json")

# NFR-01: model file up to 100 MB, dataset up to 50,000 rows
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024
MAX_DATASET_ROWS = 50_000

ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib"}
ALLOWED_DATASET_EXTENSIONS = {".csv"}

for d in (MODELS_DIR, DATASETS_DIR):
    os.makedirs(d, exist_ok=True)
