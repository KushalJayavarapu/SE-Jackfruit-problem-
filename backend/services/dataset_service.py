"""
BL-03: Allow users to upload/select a test dataset
BL-04: Validate test dataset format and required inputs

Implements the dataset side of FR-02: format checked independently at
upload, target variable specified during the upload, and the 50,000-row
limit from NFR-01. BL-04 goes a layer deeper than the upload-time check:
it looks at missing values, per-column types, and whether the target
column is actually usable for evaluation.
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
        "validation_status": "not_validated",  # set by validate_dataset()
        "validation_message": None,
        "schema_summary": None,
    }
    storage.add_record(DATASETS_INDEX, record)
    return record


def validate_dataset(dataset_id):
    """
    BL-04: re-read the stored dataset and check it is actually usable for
    evaluation — not just parseable (that was checked at upload in
    upload_dataset / FR-02). Reports per-column type/missing-value info
    so an Analyst can see what the dataset actually looks like, and flags
    the dataset invalid if the target column itself isn't usable.
    """
    record = storage.get_record(DATASETS_INDEX, dataset_id)
    if record is None:
        raise ValidationError("No such dataset upload.")

    stored_path = os.path.join(DATASETS_DIR, record["stored_filename"])

    try:
        df = pd.read_csv(stored_path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear reason
        updated = storage.update_record(DATASETS_INDEX, dataset_id, {
            "validation_status": "invalid",
            "validation_message": f"Could not re-read the dataset file: {exc}",
            "schema_summary": None,
        })
        return updated

    target_variable = record["target_variable"]
    if target_variable not in df.columns:
        updated = storage.update_record(DATASETS_INDEX, dataset_id, {
            "validation_status": "invalid",
            "validation_message": f"Target variable '{target_variable}' is missing from the file.",
            "schema_summary": None,
        })
        return updated

    target_missing = int(df[target_variable].isna().sum())
    if target_missing > 0:
        updated = storage.update_record(DATASETS_INDEX, dataset_id, {
            "validation_status": "invalid",
            "validation_message": (
                f"Target variable '{target_variable}' has {target_missing} missing "
                f"value(s); every row needs a target value to evaluate against."
            ),
            "schema_summary": None,
        })
        return updated

    columns_summary = []
    for col in df.columns:
        columns_summary.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "is_target": col == target_variable,
        })

    feature_columns = [c for c in df.columns if c != target_variable]
    if not feature_columns:
        updated = storage.update_record(DATASETS_INDEX, dataset_id, {
            "validation_status": "invalid",
            "validation_message": "Dataset has no feature columns besides the target variable.",
            "schema_summary": None,
        })
        return updated

    schema_summary = {
        "row_count": len(df),
        "feature_columns": feature_columns,
        "target_variable": target_variable,
        "columns": columns_summary,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    updated = storage.update_record(DATASETS_INDEX, dataset_id, {
        "validation_status": "valid",
        "validation_message": "Dataset re-read successfully; target variable has no missing values.",
        "schema_summary": schema_summary,
    })
    return updated


def get_dataset(dataset_id):
    return storage.get_record(DATASETS_INDEX, dataset_id)


def list_datasets():
    return storage.list_records(DATASETS_INDEX)
