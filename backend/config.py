import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

MODELS_DIR = os.path.join(STORAGE_DIR, "models")
MODELS_INDEX = os.path.join(STORAGE_DIR, "models.json")

# NFR-01: model file up to 100 MB
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024
ALLOWED_MODEL_EXTENSIONS = {".pkl", ".joblib"}

os.makedirs(MODELS_DIR, exist_ok=True)
