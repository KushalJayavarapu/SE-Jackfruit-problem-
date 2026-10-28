"""Unit/integration tests for task selection and evaluation (BL-05) — UC-05 and UC-06."""
import io
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from conftest import csv_bytes, pickle_bytes, upload
from services import dataset_service, evaluation_service, model_service
from services.model_service import ValidationError


def evaluate(model, dataset, task):
    return evaluation_service.evaluate_model(model["id"], dataset["id"], task)


# ---------------- 06_UT_03: metrics are calculated correctly ----------------
# The constant models always predict the same value, so every expected number
# below can be worked out by hand.

def test_06_UT_03_classification_metrics_match_hand_calculation(
    constant_classifier, add_valid_model, add_valid_dataset
):
    # Model always predicts 1; true labels are [1, 1, 0, 1].
    model = add_valid_model(constant_classifier())
    dataset = add_valid_dataset(csv_bytes(pd.DataFrame({"x": [0.0, 1, 2, 3], "label": [1, 1, 0, 1]})))

    metrics = evaluate(model, dataset, "classification")["metrics"]

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.5625)      # (1*0 + 3*0.75) / 4
    assert metrics["recall"] == pytest.approx(0.75)           # (1*0 + 3*1.0) / 4
    assert metrics["f1_score"] == pytest.approx(0.75 * 6 / 7)  # 3/4 * (2*.75*1/1.75)
    assert metrics["roc_auc"] == pytest.approx(0.5)           # constant scores


def test_06_UT_03_regression_metrics_match_hand_calculation(
    constant_regressor, add_valid_model, add_valid_dataset
):
    # Model always predicts 2.0; true values are [1, 2, 3, 4].
    model = add_valid_model(constant_regressor())
    dataset = add_valid_dataset(csv_bytes(pd.DataFrame({"x": [0.0, 1, 2, 3], "label": [1.0, 2, 3, 4]})))

    metrics = evaluate(model, dataset, "regression")["metrics"]

    assert metrics["mae"] == pytest.approx(1.0)               # (1+0+1+2) / 4
    assert metrics["mse"] == pytest.approx(1.5)               # (1+0+1+4) / 4
    assert metrics["rmse"] == pytest.approx(math.sqrt(1.5))
    assert metrics["r2"] == pytest.approx(-0.2)               # 1 - 6/5


def test_classification_result_contains_exactly_the_defined_metrics(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    result = evaluate(add_valid_model(classifier_bytes), add_valid_dataset(classification_csv), "classification")
    assert set(result["metrics"]) == {"accuracy", "precision", "recall", "f1_score", "roc_auc"}


def test_regression_result_contains_exactly_the_defined_metrics(
    regressor_bytes, regression_csv, add_valid_model, add_valid_dataset
):
    result = evaluate(add_valid_model(regressor_bytes), add_valid_dataset(regression_csv), "regression")
    assert set(result["metrics"]) == {"mae", "mse", "rmse", "r2"}


def test_perfect_model_scores_perfectly(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    metrics = evaluate(add_valid_model(classifier_bytes), add_valid_dataset(classification_csv), "classification")["metrics"]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_skipped_with_a_note_when_model_has_no_probabilities(add_valid_model, add_valid_dataset):
    X = pd.DataFrame({"x": [0.0, 1, 2, 3, 4, 5]})
    y = [0, 0, 0, 1, 1, 1]
    model = add_valid_model(pickle_bytes(LinearSVC().fit(X, y)))   # no predict_proba
    dataset = add_valid_dataset(csv_bytes(X.assign(label=y)))

    result = evaluate(model, dataset, "classification")

    assert result["metrics"]["roc_auc"] is None
    assert any("predict_proba" in note for note in result["notes"])
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)


def test_multiclass_classification_reports_roc_auc(add_valid_model, add_valid_dataset):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=90), "b": rng.normal(size=90)})
    y = np.repeat([0, 1, 2], 30)
    X["a"] += y * 3  # make the classes separable
    model = add_valid_model(pickle_bytes(LogisticRegression(max_iter=500).fit(X, y)))
    dataset = add_valid_dataset(csv_bytes(X.assign(label=y)))

    metrics = evaluate(model, dataset, "classification")["metrics"]
    assert metrics["roc_auc"] > 0.9


# ---------------- 05_UT_02 / 05_UT_03: task type selection ----------------

def test_05_UT_02_classification_task_is_accepted(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    result = evaluate(add_valid_model(classifier_bytes), add_valid_dataset(classification_csv), "classification")
    assert result["task_type"] == "classification"


def test_05_UT_03_regression_task_is_accepted(
    regressor_bytes, regression_csv, add_valid_model, add_valid_dataset
):
    result = evaluate(add_valid_model(regressor_bytes), add_valid_dataset(regression_csv), "regression")
    assert result["task_type"] == "regression"


@pytest.mark.parametrize("task", ["clustering", "", "Classification", None])
def test_unknown_task_type_is_rejected(
    task, classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    model = add_valid_model(classifier_bytes)
    dataset = add_valid_dataset(classification_csv)
    with pytest.raises(ValidationError, match="Task type must be"):
        evaluation_service.evaluate_model(model["id"], dataset["id"], task)


# ---------------- 05_IT_03 / 05_ST_02: task must match model and data ----------------

def test_05_IT_03_regression_task_with_a_classifier_is_rejected(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    with pytest.raises(ValidationError, match="looks like a classifier"):
        evaluate(add_valid_model(classifier_bytes), add_valid_dataset(classification_csv), "regression")


def test_05_IT_03_classification_task_with_a_regressor_is_rejected(
    regressor_bytes, regression_csv, add_valid_model, add_valid_dataset
):
    with pytest.raises(ValidationError, match="looks like a regressor"):
        evaluate(add_valid_model(regressor_bytes), add_valid_dataset(regression_csv), "classification")


def test_05_ST_02_classification_task_with_a_continuous_target_is_rejected(
    classifier_bytes, regression_csv, add_valid_model, add_valid_dataset
):
    with pytest.raises(ValidationError, match="target looks continuous"):
        evaluate(add_valid_model(classifier_bytes), add_valid_dataset(regression_csv), "classification")


def test_regression_task_with_a_text_target_is_rejected(
    regressor_bytes, add_valid_model, add_valid_dataset
):
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
    with pytest.raises(ValidationError, match="not numeric"):
        evaluate(add_valid_model(regressor_bytes), add_valid_dataset(csv_bytes(frame)), "regression")


# ---------------- 04_IT_02 / 04_ST_02: model/dataset compatibility ----------------

def test_04_IT_02_model_with_features_missing_from_dataset_is_rejected(
    classifier_bytes, add_valid_model, add_valid_dataset
):
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "other": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    with pytest.raises(ValidationError, match="not found in the dataset: f2"):
        evaluate(add_valid_model(classifier_bytes), add_valid_dataset(csv_bytes(frame)), "classification")


def test_04_ST_02_model_with_wrong_feature_count_is_rejected(add_valid_model, add_valid_dataset):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))                        # fitted on 3 unnamed features
    y = (X[:, 0] > 0).astype(int)
    model = add_valid_model(pickle_bytes(LogisticRegression().fit(X, y)))
    two_features = pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "label": y})
    dataset = add_valid_dataset(csv_bytes(two_features))

    with pytest.raises(ValidationError, match="expects 3 feature"):
        evaluate(model, dataset, "classification")


def test_04_IT_02_compatible_model_is_accepted(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    result = evaluate(add_valid_model(classifier_bytes), add_valid_dataset(classification_csv), "classification")
    assert result["notes"] == []


def test_extra_dataset_columns_are_ignored_and_reported(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    frame = pd.read_csv(io.BytesIO(classification_csv)).assign(row_id=range(60))
    result = evaluate(add_valid_model(classifier_bytes), add_valid_dataset(csv_bytes(frame)), "classification")

    assert any("row_id" in note for note in result["notes"])


def test_dataset_column_order_does_not_matter(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    shuffled = pd.read_csv(io.BytesIO(classification_csv))[["label", "f2", "f1"]]
    result = evaluate(add_valid_model(classifier_bytes), add_valid_dataset(csv_bytes(shuffled)), "classification")
    assert result["metrics"]["accuracy"] == pytest.approx(1.0)


# ---------------- inputs must be validated first ----------------

def test_unvalidated_model_cannot_be_evaluated(classifier_bytes, classification_csv, add_valid_dataset):
    model = model_service.upload_model(upload(classifier_bytes, "m.pkl"))   # never validated
    with pytest.raises(ValidationError, match="model has not passed validation"):
        evaluate(model, add_valid_dataset(classification_csv), "classification")


def test_unvalidated_dataset_cannot_be_evaluated(classifier_bytes, classification_csv, add_valid_model):
    dataset = dataset_service.upload_dataset(upload(classification_csv, "d.csv"), "label")  # never validated
    with pytest.raises(ValidationError, match="dataset has not passed validation"):
        evaluate(add_valid_model(classifier_bytes), dataset, "classification")


def test_model_that_failed_validation_cannot_be_evaluated(classification_csv, add_valid_model, add_valid_dataset):
    corrupt = add_valid_model(b"not a pickle")
    assert corrupt["validation_status"] == "invalid"
    with pytest.raises(ValidationError, match="model has not passed validation"):
        evaluate(corrupt, add_valid_dataset(classification_csv), "classification")


def test_unknown_model_and_dataset_ids_are_rejected(classifier_bytes, classification_csv, add_valid_model, add_valid_dataset):
    model = add_valid_model(classifier_bytes)
    dataset = add_valid_dataset(classification_csv)
    with pytest.raises(ValidationError, match="No such model"):
        evaluation_service.evaluate_model("nope", dataset["id"], "classification")
    with pytest.raises(ValidationError, match="No such dataset"):
        evaluation_service.evaluate_model(model["id"], "nope", "classification")


# ---------------- 06_IT_03: results are stored ----------------

def test_06_IT_03_completed_evaluation_is_stored_with_its_metadata(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    model = add_valid_model(classifier_bytes, "my_model.pkl")
    dataset = add_valid_dataset(classification_csv, filename="my_data.csv")

    result = evaluate(model, dataset, "classification")

    assert result["model_id"] == model["id"]
    assert result["dataset_id"] == dataset["id"]
    assert result["model_filename"] == "my_model.pkl"
    assert result["dataset_filename"] == "my_data.csv"
    assert result["task_type"] == "classification"
    assert result["row_count"] == 60
    assert result["evaluated_at"].endswith("Z")
    assert result["duration_seconds"] >= 0
    assert evaluation_service.list_evaluations() == [result]


def test_running_the_same_evaluation_twice_stores_two_records(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    model = add_valid_model(classifier_bytes)
    dataset = add_valid_dataset(classification_csv)
    first = evaluate(model, dataset, "classification")
    second = evaluate(model, dataset, "classification")

    assert first["id"] != second["id"]
    assert len(evaluation_service.list_evaluations()) == 2


# ---------------- 06_ST_02: failures leave nothing behind ----------------

def test_06_ST_02_rejected_evaluation_stores_nothing(
    classifier_bytes, classification_csv, add_valid_model, add_valid_dataset
):
    model = add_valid_model(classifier_bytes)
    dataset = add_valid_dataset(classification_csv)

    with pytest.raises(ValidationError):
        evaluate(model, dataset, "regression")        # task mismatch

    assert evaluation_service.list_evaluations() == []


def test_06_ST_02_model_that_crashes_while_predicting_gives_a_clear_error_and_stores_nothing(
    classifier_bytes, add_valid_model, add_valid_dataset
):
    # Passes validation (columns exist) but f1 holds text, so predict() fails.
    frame = pd.DataFrame({"f1": ["a", "b", "c"], "f2": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    model = add_valid_model(classifier_bytes)
    dataset = add_valid_dataset(csv_bytes(frame))

    with pytest.raises(ValidationError, match="Evaluation failed while running the model"):
        evaluate(model, dataset, "classification")

    assert evaluation_service.list_evaluations() == []
