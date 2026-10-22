"""
BL-03: Allow users to upload/select a test dataset

Implements the dataset side of FR-02: format checked independently at
upload, target variable specified during the upload, and the 50,000-row
limit from NFR-01.
"""
import os
import uuid
import datetime

import pandas as pd

from config import DATASETS_DIR, DATASETS_INDEX, ALLOWED_DATASET_EXTENSIONS, MAX_DATASET_ROWS
from services import storage
from services.model_service import ValidationError  # same error type, shared meaning


def _extension_of(filename):
    return os.path.splitext(filename)[1].lower()


def upload_dataset(file_storage, target_variable):
    if not file_storage or file_storage.filename == "":
        raise ValidationError("No file was selected.")

    if not target_variable or not target_variable.strip():
        raise ValidationError("A target variable must be specified for this dataset.")
    target_variable = target_variable.strip()

    original_filename = file_storage.filename
    ext = _extension_of(original_filename)
    if ext not in ALLOWED_DATASET_EXTENSIONS:
        raise ValidationError(
            f"Unsupported dataset file type '{ext or 'unknown'}'. "
            f"Accepted types: {', '.join(sorted(ALLOWED_DATASET_EXTENSIONS))}."
        )

    try:
        df = pd.read_csv(file_storage.stream)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear reason (FR-02)
        raise ValidationError(f"Could not read the dataset file: {exc}")

    if df.empty:
        raise ValidationError("The dataset file has no rows.")

    row_count = len(df)
    if row_count > MAX_DATASET_ROWS:
        raise ValidationError(
            f"Dataset has {row_count} rows, exceeding the {MAX_DATASET_ROWS} row limit."
        )

    if target_variable not in df.columns:
        raise ValidationError(
            f"Target variable '{target_variable}' is not a column in this dataset. "
            f"Available columns: {', '.join(df.columns)}."
        )

    dataset_id = str(uuid.uuid4())
    stored_filename = f"{dataset_id}{ext}"
    stored_path = os.path.join(DATASETS_DIR, stored_filename)
    df.to_csv(stored_path, index=False)

    record = {
        "id": dataset_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "row_count": row_count,
        "columns": list(df.columns),
        "target_variable": target_variable,
        "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "format_checked": True,
    }
    storage.add_record(DATASETS_INDEX, record)
    return record


def get_dataset(dataset_id):
    return storage.get_record(DATASETS_INDEX, dataset_id)


def list_datasets():
    return storage.list_records(DATASETS_INDEX)
