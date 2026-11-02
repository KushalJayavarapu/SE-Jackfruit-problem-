"""
BL-01: upload a model file.  BL-02: validate it.
Every model belongs to the account that uploaded it (NFR-02).
"""
import os
import uuid

import joblib

import config
from services import compatibility, db
from services.errors import NotFoundError, ValidationError  # noqa: F401 (re-exported)
from services.timeutil import now_iso


def _extension_of(filename):
    return os.path.splitext(filename)[1].lower()


def upload_model(file_storage, owner_id):
    """Save an uploaded model after format and size checks (FR-02, NFR-01)."""
    if not file_storage or file_storage.filename == "":
        raise ValidationError("No file was selected.")

    original_filename = os.path.basename(file_storage.filename)
    ext = _extension_of(original_filename)
    if ext not in config.ALLOWED_MODEL_EXTENSIONS:
        raise ValidationError(
            f"Unsupported model file type '{ext or 'unknown'}'. "
            f"Accepted types: {', '.join(sorted(config.ALLOWED_MODEL_EXTENSIONS))}."
        )

    file_storage.stream.seek(0, os.SEEK_END)
    size_bytes = file_storage.stream.tell()
    file_storage.stream.seek(0)

    if size_bytes == 0:
        raise ValidationError("The uploaded file is empty.")
    if size_bytes > config.MAX_MODEL_SIZE_BYTES:
        mb = config.MAX_MODEL_SIZE_BYTES / (1024 * 1024)
        raise ValidationError(f"Model file exceeds the {mb:.0f} MB limit.")

    model_id = str(uuid.uuid4())
    stored_filename = f"{model_id}{ext}"
    file_storage.save(os.path.join(config.MODELS_DIR, stored_filename))

    record = {
        "id": model_id,
        "owner_id": owner_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "size_bytes": size_bytes,
        "uploaded_at": now_iso(),
        "validation_status": "not_validated",
        "validation_message": None,
        "required_inputs": None,
    }
    db.insert("models", record)
    return record


def get_model(model_id, owner_id):
    """The model, or None if it does not exist or belongs to someone else."""
    return db.fetch_one("models", "id = ? AND owner_id = ?", (model_id, owner_id))


def list_models(owner_id):
    return db.fetch_all("models", "owner_id = ?", (owner_id,))


def _set_validation(model_id, status, message, required_inputs=None):
    db.update("models", model_id, {
        "validation_status": status,
        "validation_message": message,
        "required_inputs": required_inputs,
    })


def validate_model(model_id, owner_id):
    """
    Load the model, confirm it has predict(), record the inputs it expects, and
    report which of the owner's validated datasets it can be evaluated on (UC-04).
    """
    record = get_model(model_id, owner_id)
    if record is None:
        raise NotFoundError("No such model upload.")

    try:
        model_obj = joblib.load(os.path.join(config.MODELS_DIR, record["stored_filename"]))
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a clear reason
        _set_validation(model_id, "invalid", f"Could not load the model file: {exc}")
        return get_model(model_id, owner_id)

    if not hasattr(model_obj, "predict"):
        _set_validation(model_id, "invalid",
                        "File loaded but is not a usable model (no predict() method found).")
        return get_model(model_id, owner_id)

    names = getattr(model_obj, "feature_names_in_", None)
    required_inputs = {
        "n_features_in": int(model_obj.n_features_in_) if hasattr(model_obj, "n_features_in_") else None,
        "feature_names_in": [str(n) for n in names] if names is not None else None,
        "model_type": type(model_obj).__name__,
        "is_classifier": hasattr(model_obj, "classes_"),
        "has_predict_proba": hasattr(model_obj, "predict_proba"),
    }
    _set_validation(model_id, "valid", "Model loaded successfully and exposes predict().", required_inputs)

    result = get_model(model_id, owner_id)
    result["compatibility"] = dataset_compatibility(result, owner_id)
    return result


def dataset_compatibility(model_record, owner_id):
    """Check the model against each of the owner's validated datasets (UC-04)."""
    datasets = db.fetch_all("datasets", "owner_id = ? AND validation_status = 'valid'", (owner_id,))
    report = []
    for dataset in datasets:
        ok, reason = compatibility.check_records(model_record, dataset)
        report.append({
            "dataset_id": dataset["id"],
            "dataset_filename": dataset["original_filename"],
            "compatible": ok,
            "reason": reason,
        })
    return report


def delete_model(model_id, owner_id):
    record = get_model(model_id, owner_id)
    if record is None:
        raise NotFoundError("No such model upload.")
    path = os.path.join(config.MODELS_DIR, record["stored_filename"])
    if os.path.exists(path):
        os.remove(path)
    db.delete("models", model_id)
