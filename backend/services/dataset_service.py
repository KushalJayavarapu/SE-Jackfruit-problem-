"""
BL-03: upload a dataset with its target column.  BL-04: validate it.
Every dataset belongs to the account that uploaded it (NFR-02).
"""
import os
import uuid

import pandas as pd

import config
from services import compatibility, db
from services.errors import NotFoundError, ValidationError
from services.timeutil import now_iso


def _extension_of(filename):
    return os.path.splitext(filename)[1].lower()


def upload_dataset(file_storage, target_variable, owner_id):
    if not file_storage or file_storage.filename == "":
        raise ValidationError("No file was selected.")

    if not target_variable or not target_variable.strip():
        raise ValidationError("A target variable must be specified for this dataset.")
    target_variable = target_variable.strip()

    original_filename = os.path.basename(file_storage.filename)
    ext = _extension_of(original_filename)
    if ext not in config.ALLOWED_DATASET_EXTENSIONS:
        raise ValidationError(
            f"Unsupported dataset file type '{ext or 'unknown'}'. "
            f"Accepted types: {', '.join(sorted(config.ALLOWED_DATASET_EXTENSIONS))}."
        )

    try:
        df = pd.read_csv(file_storage.stream)
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear reason (FR-02)
        raise ValidationError(f"Could not read the dataset file: {exc}")

    if df.empty:
        raise ValidationError("The dataset file has no rows.")

    row_count = len(df)
    if row_count > config.MAX_DATASET_ROWS:
        raise ValidationError(
            f"Dataset has {row_count} rows, exceeding the {config.MAX_DATASET_ROWS} row limit."
        )

    columns = [str(c) for c in df.columns]
    if target_variable not in columns:
        raise ValidationError(
            f"Target variable '{target_variable}' is not a column in this dataset. "
            f"Available columns: {', '.join(columns)}."
        )

    dataset_id = str(uuid.uuid4())
    stored_filename = f"{dataset_id}{ext}"
    df.to_csv(os.path.join(config.DATASETS_DIR, stored_filename), index=False)

    record = {
        "id": dataset_id,
        "owner_id": owner_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "row_count": row_count,
        "columns": columns,
        "target_variable": target_variable,
        "uploaded_at": now_iso(),
        "validation_status": "not_validated",
        "validation_message": None,
        "schema_summary": None,
    }
    db.insert("datasets", record)
    return record


def get_dataset(dataset_id, owner_id):
    return db.fetch_one("datasets", "id = ? AND owner_id = ?", (dataset_id, owner_id))


def list_datasets(owner_id):
    return db.fetch_all("datasets", "owner_id = ?", (owner_id,))


def _set_validation(dataset_id, status, message, schema_summary=None):
    db.update("datasets", dataset_id, {
        "validation_status": status,
        "validation_message": message,
        "schema_summary": schema_summary,
    })


def validate_dataset(dataset_id, owner_id):
    """
    Re-read the stored file and check it is usable for evaluation: the target
    has no missing values and there is at least one feature column.
    """
    record = get_dataset(dataset_id, owner_id)
    if record is None:
        raise NotFoundError("No such dataset upload.")

    try:
        df = pd.read_csv(os.path.join(config.DATASETS_DIR, record["stored_filename"]))
    except Exception as exc:  # noqa: BLE001
        _set_validation(dataset_id, "invalid", f"Could not re-read the dataset file: {exc}")
        return get_dataset(dataset_id, owner_id)

    target = record["target_variable"]
    if target not in df.columns:
        _set_validation(dataset_id, "invalid", f"Target variable '{target}' is missing from the file.")
        return get_dataset(dataset_id, owner_id)

    target_missing = int(df[target].isna().sum())
    if target_missing > 0:
        _set_validation(dataset_id, "invalid", (
            f"Target variable '{target}' has {target_missing} missing "
            f"value(s); every row needs a target value to evaluate against."
        ))
        return get_dataset(dataset_id, owner_id)

    feature_columns = [c for c in df.columns if c != target]
    if not feature_columns:
        _set_validation(dataset_id, "invalid", "Dataset has no feature columns besides the target variable.")
        return get_dataset(dataset_id, owner_id)

    schema_summary = {
        "row_count": len(df),
        "feature_columns": feature_columns,
        "target_variable": target,
        "columns": [{
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "is_target": col == target,
        } for col in df.columns],
        "duplicate_rows": int(df.duplicated().sum()),
    }
    _set_validation(dataset_id, "valid",
                    "Dataset re-read successfully; target variable has no missing values.", schema_summary)

    result = get_dataset(dataset_id, owner_id)
    result["compatibility"] = model_compatibility(result, owner_id)
    return result


def model_compatibility(dataset_record, owner_id):
    """Check the dataset against each of the owner's validated models."""
    models = db.fetch_all("models", "owner_id = ? AND validation_status = 'valid'", (owner_id,))
    report = []
    for model in models:
        ok, reason = compatibility.check_records(model, dataset_record)
        report.append({
            "model_id": model["id"],
            "model_filename": model["original_filename"],
            "compatible": ok,
            "reason": reason,
        })
    return report


def delete_dataset(dataset_id, owner_id):
    record = get_dataset(dataset_id, owner_id)
    if record is None:
        raise NotFoundError("No such dataset upload.")
    path = os.path.join(config.DATASETS_DIR, record["stored_filename"])
    if os.path.exists(path):
        os.remove(path)
    db.delete("datasets", dataset_id)
