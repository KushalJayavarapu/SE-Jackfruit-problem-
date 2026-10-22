"""
BL-01: Upload a machine learning model through the web interface
BL-02: Validate uploaded model format and required inputs

Covers the model side of FR-02 (per-file format check at upload time) plus
the model-validation half of what FR-02/FR-03 need before evaluation can
run. Compatibility against a specific dataset (the other half of FR-02) is
out of scope here — that's BL-04.
"""
import os
import uuid
import datetime

import joblib

from config import MODELS_DIR, MODELS_INDEX, MAX_MODEL_SIZE_BYTES, ALLOWED_MODEL_EXTENSIONS
from services import storage


class ValidationError(Exception):
    """Raised with a clear, user-facing reason a file was rejected (FR-02)."""


def _extension_of(filename):
    return os.path.splitext(filename)[1].lower()


def upload_model(file_storage):
    """
    Save an uploaded model file after format/size checks.
    `file_storage` is a werkzeug FileStorage (Flask's request.files[...]).
    Returns the stored record on success; raises ValidationError on a
    rejected file, with a message meant to be shown to the Analyst.
    """
    if not file_storage or file_storage.filename == "":
        raise ValidationError("No file was selected.")

    original_filename = file_storage.filename
    ext = _extension_of(original_filename)
    if ext not in ALLOWED_MODEL_EXTENSIONS:
        raise ValidationError(
            f"Unsupported model file type '{ext or 'unknown'}'. "
            f"Accepted types: {', '.join(sorted(ALLOWED_MODEL_EXTENSIONS))}."
        )

    # Read into memory once to check size before writing to disk.
    file_storage.stream.seek(0, os.SEEK_END)
    size_bytes = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size_bytes == 0:
        raise ValidationError("The uploaded file is empty.")
    if size_bytes > MAX_MODEL_SIZE_BYTES:
        mb = MAX_MODEL_SIZE_BYTES / (1024 * 1024)
        raise ValidationError(f"Model file exceeds the {mb:.0f} MB limit.")

    model_id = str(uuid.uuid4())
    stored_filename = f"{model_id}{ext}"
    stored_path = os.path.join(MODELS_DIR, stored_filename)
    file_storage.save(stored_path)

    record = {
        "id": model_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "size_bytes": size_bytes,
        "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "format_checked": True,
        "validation_status": "not_validated",  # set by validate_model()
        "validation_message": None,
        "required_inputs": None,
    }
    storage.add_record(MODELS_INDEX, record)
    return record


def validate_model(model_id):
    """
    BL-02: load the model and confirm it is actually usable, and extract
    the inputs it expects so the Analyst (or FR-02's later compatibility
    check) knows what the dataset needs to provide.
    """
    record = storage.get_record(MODELS_INDEX, model_id)
    if record is None:
        raise ValidationError("No such model upload.")

    stored_path = os.path.join(MODELS_DIR, record["stored_filename"])

    try:
        model_obj = joblib.load(stored_path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a clear reason
        updated = storage.update_record(MODELS_INDEX, model_id, {
            "validation_status": "invalid",
            "validation_message": f"Could not load the model file: {exc}",
            "required_inputs": None,
        })
        return updated

    if not hasattr(model_obj, "predict"):
        updated = storage.update_record(MODELS_INDEX, model_id, {
            "validation_status": "invalid",
            "validation_message": "File loaded but is not a usable model "
                                   "(no predict() method found).",
            "required_inputs": None,
        })
        return updated

    required_inputs = {
        "n_features_in": getattr(model_obj, "n_features_in_", None),
        "feature_names_in": (
            list(getattr(model_obj, "feature_names_in_"))
            if hasattr(model_obj, "feature_names_in_")
            else None
        ),
        "model_type": type(model_obj).__name__,
    }

    updated = storage.update_record(MODELS_INDEX, model_id, {
        "validation_status": "valid",
        "validation_message": "Model loaded successfully and exposes predict().",
        "required_inputs": required_inputs,
    })
    return updated


def get_model(model_id):
    return storage.get_record(MODELS_INDEX, model_id)


def list_models():
    return storage.list_records(MODELS_INDEX)
