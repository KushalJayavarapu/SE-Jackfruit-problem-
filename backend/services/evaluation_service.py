"""
BL-05: Evaluate a machine learning model against a test dataset

Runs a validated model on a validated dataset and computes the metrics
for the declared task type (FR-04):
  classification -> accuracy, precision, recall, F1, ROC-AUC
  regression     -> MAE, MSE, RMSE, R2

Before running, it checks the model/dataset/task are compatible (back half
of FR-02 and FR-03) and rejects mismatches with a specific reason (NFR-03).
"""
import os
import time
import uuid
import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from config import (
    MODELS_DIR, DATASETS_DIR, MODELS_INDEX, DATASETS_INDEX, EVALUATIONS_INDEX,
)
from services import storage
from services.model_service import ValidationError

TASK_TYPES = {"classification", "regression"}


def _load_valid_model(model_id):
    record = storage.get_record(MODELS_INDEX, model_id)
    if record is None:
        raise ValidationError("No such model.")
    if record.get("validation_status") != "valid":
        raise ValidationError("This model has not passed validation yet. Validate it first.")
    try:
        return record, joblib.load(os.path.join(MODELS_DIR, record["stored_filename"]))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not load the model file: {exc}")


def _load_valid_dataset(dataset_id):
    record = storage.get_record(DATASETS_INDEX, dataset_id)
    if record is None:
        raise ValidationError("No such dataset.")
    if record.get("validation_status") != "valid":
        raise ValidationError("This dataset has not passed validation yet. Validate it first.")
    try:
        return record, pd.read_csv(os.path.join(DATASETS_DIR, record["stored_filename"]))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Could not read the dataset file: {exc}")


def _check_model_dataset_compatibility(model_obj, feature_columns):
    """Model's expected inputs must line up with the dataset's feature columns (FR-02)."""
    names = getattr(model_obj, "feature_names_in_", None)
    if names is not None:
        expected = list(names)
        missing = [c for c in expected if c not in feature_columns]
        if missing:
            raise ValidationError(
                f"Model expects feature(s) not found in the dataset: {', '.join(map(str, missing))}."
            )
        return expected  # use the model's column order

    n_expected = getattr(model_obj, "n_features_in_", None)
    if n_expected is not None and n_expected != len(feature_columns):
        raise ValidationError(
            f"Model expects {n_expected} feature(s) but the dataset has "
            f"{len(feature_columns)} (excluding the target)."
        )
    return feature_columns


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


def _classification_metrics(model_obj, X, y, y_pred):
    result = {
        "accuracy": float(skm.accuracy_score(y, y_pred)),
        "precision": float(skm.precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall": float(skm.recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(skm.f1_score(y, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
    }
    notes = []
    if hasattr(model_obj, "predict_proba"):
        try:
            proba = model_obj.predict_proba(X)
            classes = list(model_obj.classes_)
            if len(classes) == 2:
                result["roc_auc"] = float(skm.roc_auc_score(y, proba[:, 1]))
            else:
                result["roc_auc"] = float(skm.roc_auc_score(
                    y, proba, multi_class="ovr", average="weighted", labels=classes))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"ROC-AUC could not be computed: {exc}")
    else:
        notes.append("ROC-AUC not available: model has no predict_proba().")
    return result, notes


def _regression_metrics(y, y_pred):
    mse = float(skm.mean_squared_error(y, y_pred))
    return {
        "mae": float(skm.mean_absolute_error(y, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(skm.r2_score(y, y_pred)),
    }, []


def evaluate_model(model_id, dataset_id, task_type):
    if task_type not in TASK_TYPES:
        raise ValidationError("Task type must be 'classification' or 'regression'.")

    model_record, model_obj = _load_valid_model(model_id)
    dataset_record, df = _load_valid_dataset(dataset_id)

    target = dataset_record["target_variable"]
    feature_columns = [c for c in df.columns if c != target]
    ordered_features = _check_model_dataset_compatibility(model_obj, feature_columns)
    ignored = [c for c in feature_columns if c not in ordered_features]

    X = df[ordered_features]
    y = df[target]
    _check_task_compatibility(task_type, model_obj, y)

    started = time.perf_counter()
    try:
        y_pred = model_obj.predict(X)
        if task_type == "classification":
            metric_values, notes = _classification_metrics(model_obj, X, y, y_pred)
        else:
            metric_values, notes = _regression_metrics(y, y_pred)
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - NFR-03: specific reason, no silent failure
        raise ValidationError(f"Evaluation failed while running the model: {exc}")
    duration = time.perf_counter() - started
    if ignored:
        notes.append(f"Dataset column(s) not used by the model were ignored: {', '.join(map(str, ignored))}.")

    record = {
        "id": str(uuid.uuid4()),
        "model_id": model_id,
        "model_filename": model_record["original_filename"],
        "dataset_id": dataset_id,
        "dataset_filename": dataset_record["original_filename"],
        "task_type": task_type,
        "row_count": len(df),
        "metrics": metric_values,
        "notes": notes,
        "duration_seconds": round(duration, 4),
        "evaluated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    storage.add_record(EVALUATIONS_INDEX, record)
    return record


def list_evaluations():
    return storage.list_records(EVALUATIONS_INDEX)
