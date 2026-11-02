"""
BL-05 / FR-03 / FR-04 / UC-06: evaluate a model on a dataset.
Also: reopen (UC-07/UC-10) and compare (FR-06/UC-09) saved evaluations.

Before running, the model, dataset and task type are checked against each
other and a mismatch is rejected with a reason (NFR-03). If anything fails
during prediction, metrics or charts, nothing is stored (UC-06 alternative flow).
"""
import os
import time
import uuid

import joblib
import pandas as pd

import config
from services import compatibility, db, metrics, visualization_service
from services.errors import NotFoundError, ValidationError
from services.timeutil import now_iso

TASK_TYPES = set(metrics.TASKS)


def _load_valid_model(model_id, owner_id):
    record = db.fetch_one("models", "id = ? AND owner_id = ?", (model_id, owner_id))
    if record is None:
        raise ValidationError("No such model.")
    if record.get("validation_status") != "valid":
        raise ValidationError("This model has not passed validation yet. Validate it first.")
    try:
        return record, joblib.load(os.path.join(config.MODELS_DIR, record["stored_filename"]))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not load the model file: {exc}")


def _load_valid_dataset(dataset_id, owner_id):
    record = db.fetch_one("datasets", "id = ? AND owner_id = ?", (dataset_id, owner_id))
    if record is None:
        raise ValidationError("No such dataset.")
    if record.get("validation_status") != "valid":
        raise ValidationError("This dataset has not passed validation yet. Validate it first.")
    try:
        return record, pd.read_csv(os.path.join(config.DATASETS_DIR, record["stored_filename"]))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not read the dataset file: {exc}")


def _check_task_compatibility(task_type, model_obj, y):
    """Declared task must match both the model and the dataset's target (FR-03)."""
    is_classifier = hasattr(model_obj, "classes_")
    if task_type == "classification" and not is_classifier:
        raise ValidationError("Task is 'classification' but the model looks like a regressor (no classes_).")
    if task_type == "regression" and is_classifier:
        raise ValidationError("Task is 'regression' but the model looks like a classifier.")
    if task_type == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValidationError("Task is 'regression' but the target column is not numeric.")
    if task_type == "classification" and pd.api.types.is_float_dtype(y) and y.nunique() > 20:
        raise ValidationError(
            "Task is 'classification' but the target looks continuous "
            f"({y.nunique()} distinct float values)."
        )


def evaluate_model(model_id, dataset_id, task_type, owner_id):
    if task_type not in TASK_TYPES:
        raise ValidationError("Task type must be 'classification' or 'regression'.")

    model_record, model_obj = _load_valid_model(model_id, owner_id)
    dataset_record, df = _load_valid_dataset(dataset_id, owner_id)

    target = dataset_record["target_variable"]
    feature_columns = [c for c in df.columns if c != target]
    names = getattr(model_obj, "feature_names_in_", None)
    ordered_features, reason = compatibility.check(
        list(names) if names is not None else None,
        getattr(model_obj, "n_features_in_", None),
        feature_columns,
    )
    if ordered_features is None:
        raise ValidationError(reason)
    ignored = [c for c in feature_columns if c not in ordered_features]

    X = df[ordered_features]
    y = df[target]
    _check_task_compatibility(task_type, model_obj, y)

    started = time.perf_counter()
    try:
        y_pred = model_obj.predict(X)
        proba, classes, notes = None, None, []
        if task_type == "classification":
            classes = list(model_obj.classes_)
            if hasattr(model_obj, "predict_proba"):
                try:
                    proba = model_obj.predict_proba(X)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"ROC-AUC not available: predict_proba() failed ({exc}).")
        metric_values, metric_notes = metrics.compute(task_type, y, y_pred, proba, classes)
        if not notes:
            notes.extend(metric_notes)
        else:
            metric_values["roc_auc"] = None
        chart_data = visualization_service.build_chart_data(
            task_type, y, y_pred, proba, classes,
            roc_available=metric_values.get("roc_auc") is not None,
        )
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - NFR-03: specific reason, nothing stored
        raise ValidationError(f"Evaluation failed while running the model: {exc}")
    duration = time.perf_counter() - started

    if ignored:
        notes.append(f"Dataset column(s) not used by the model were ignored: {', '.join(map(str, ignored))}.")
    if task_type == "regression" and chart_data["residuals"]["sampled"]:
        notes.append(
            f"Charts show a random sample of {config.MAX_RESIDUAL_POINTS:,} rows; "
            "metrics use every row."
        )

    record = {
        "id": str(uuid.uuid4()),
        "owner_id": owner_id,
        "model_id": model_id,
        "model_filename": model_record["original_filename"],
        "model_type": (model_record.get("required_inputs") or {}).get("model_type"),
        "dataset_id": dataset_id,
        "dataset_filename": dataset_record["original_filename"],
        "target_variable": target,
        "task_type": task_type,
        "row_count": len(df),
        "metrics": metric_values,
        "notes": notes,
        "charts": chart_data,
        "duration_seconds": round(duration, 4),
        "evaluated_at": now_iso(),
    }
    db.insert("evaluations", record)
    return public_evaluation(record)


def public_evaluation(record):
    """What the API returns: the record without raw chart data, plus the chart list."""
    if record is None:
        return None
    result = {k: v for k, v in record.items() if k != "charts"}
    result["charts"] = visualization_service.available_charts(record.get("charts"))
    return result


def get_evaluation_record(evaluation_id, owner_id):
    """Full stored record, including chart data, or None (NFR-02)."""
    return db.fetch_one("evaluations", "id = ? AND owner_id = ?", (evaluation_id, owner_id))


def get_evaluation(evaluation_id, owner_id):
    record = get_evaluation_record(evaluation_id, owner_id)
    if record is None:
        raise NotFoundError("This evaluation is not available. It may have been removed.")
    return public_evaluation(record)


def list_evaluations(owner_id):
    return [public_evaluation(r) for r in db.fetch_all(
        "evaluations", "owner_id = ?", (owner_id,), order="evaluated_at DESC")]


def get_many(evaluation_ids, owner_id):
    """Fetch several saved evaluations in the order given. All must exist (UC-11)."""
    ids = [i for i in dict.fromkeys(i.strip() for i in evaluation_ids) if i]
    if not ids:
        raise ValidationError("Select at least one evaluation.")
    records = [get_evaluation_record(i, owner_id) for i in ids]
    if any(r is None for r in records):
        raise NotFoundError("One or more of the selected evaluations no longer exist.")
    return records


def compare(evaluation_ids, owner_id):
    """FR-06 / UC-09: side-by-side metrics for 2+ evaluations of the same task type."""
    records = get_many(evaluation_ids, owner_id)
    if len(records) < 2:
        raise ValidationError("Select two or more evaluations to compare.")
    task_types = sorted({r["task_type"] for r in records})
    if len(task_types) > 1:
        raise ValidationError(
            "The selected evaluations have different task types "
            f"({' and '.join(task_types)}). Only evaluations of the same task type can be compared."
        )
    task_type = task_types[0]
    return {
        "task_type": task_type,
        "metric_keys": metrics.metric_keys(task_type),
        "evaluations": [public_evaluation(r) for r in records],
    }
