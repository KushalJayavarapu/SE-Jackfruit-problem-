"""
Metric definitions per task type (FR-04, DR-01, DR-02).

To add a metric, add it to the task's function below and to METRIC_INFO;
nothing in upload, storage or authentication needs to change (NFR-05).
"""
import numpy as np
from sklearn import metrics as skm

# Display name, and whether a higher value is better. Used by the UI and reports.
METRIC_INFO = {
    "accuracy": ("Accuracy", True),
    "precision": ("Precision (weighted)", True),
    "recall": ("Recall (weighted)", True),
    "f1_score": ("F1 score (weighted)", True),
    "roc_auc": ("ROC-AUC", True),
    "mae": ("MAE", False),
    "mse": ("MSE", False),
    "rmse": ("RMSE", False),
    "r2": ("R²", True),
}


def classification(y, y_pred, proba, classes):
    result = {
        "accuracy": float(skm.accuracy_score(y, y_pred)),
        "precision": float(skm.precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall": float(skm.recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(skm.f1_score(y, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
    }
    notes = []
    if proba is None:
        notes.append("ROC-AUC not available: model has no predict_proba().")
        return result, notes
    try:
        if len(classes) == 2:
            result["roc_auc"] = float(skm.roc_auc_score(y, proba[:, 1]))
        else:
            result["roc_auc"] = float(skm.roc_auc_score(
                y, proba, multi_class="ovr", average="weighted", labels=classes))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"ROC-AUC could not be computed: {exc}")
    return result, notes


def regression(y, y_pred, proba=None, classes=None):
    mse = float(skm.mean_squared_error(y, y_pred))
    return {
        "mae": float(skm.mean_absolute_error(y, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(skm.r2_score(y, y_pred)),
    }, []


TASKS = {
    "classification": classification,
    "regression": regression,
}


def compute(task_type, y, y_pred, proba, classes):
    return TASKS[task_type](y, y_pred, proba, classes)


def metric_keys(task_type):
    return {
        "classification": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        "regression": ["mae", "mse", "rmse", "r2"],
    }[task_type]
