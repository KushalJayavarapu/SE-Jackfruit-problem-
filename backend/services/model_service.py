"""
BL-01: Upload a machine learning model through the web interface

Covers the model side of FR-02 (per-file format check at upload time)
and the model size limit from NFR-01.
"""
import os
import uuid
import datetime

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
    }
    storage.add_record(MODELS_INDEX, record)
    return record


def get_model(model_id):
    return storage.get_record(MODELS_INDEX, model_id)


def list_models():
    return storage.list_records(MODELS_INDEX)
