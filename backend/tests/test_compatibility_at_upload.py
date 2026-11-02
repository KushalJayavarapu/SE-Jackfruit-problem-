"""UC-04: when a model is validated, check it against the datasets already uploaded;
and the same the other way round when a dataset is validated."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from conftest import csv_bytes, pickle_bytes
from test_api_workflow import upload_and_validate_dataset, upload_and_validate_model


def test_validated_model_reports_fit_with_each_existing_dataset(api_client, classifier_bytes, classification_csv):
    upload_and_validate_dataset(api_client, classification_csv)
    odd = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "unrelated": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
    upload_and_validate_dataset(api_client, csv_bytes(odd))

    model = upload_and_validate_model(api_client, classifier_bytes)

    assert len(model["compatibility"]) == 2
    compatible = [c for c in model["compatibility"] if c["compatible"]]
    incompatible = [c for c in model["compatibility"] if not c["compatible"]]
    assert len(compatible) == 1 and compatible[0]["reason"] is None
    assert len(incompatible) == 1 and "not found in the dataset: f2" in incompatible[0]["reason"]


def test_validated_dataset_reports_fit_with_each_existing_model(api_client, classifier_bytes, classification_csv):
    upload_and_validate_model(api_client, classifier_bytes)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    three = pickle_bytes(LogisticRegression().fit(X, (X[:, 0] > 0).astype(int)))
    upload_and_validate_model(api_client, three)

    dataset = upload_and_validate_dataset(api_client, classification_csv)

    verdicts = sorted(c["compatible"] for c in dataset["compatibility"])
    assert verdicts == [False, True]
    reason = next(c["reason"] for c in dataset["compatibility"] if not c["compatible"])
    assert "expects 3 feature" in reason


def test_no_compatibility_entries_when_nothing_to_compare_with(api_client, classifier_bytes):
    assert upload_and_validate_model(api_client, classifier_bytes)["compatibility"] == []


def test_invalid_datasets_are_left_out_of_the_compatibility_report(api_client, classifier_bytes):
    bad = pd.DataFrame({"f1": [1.0, 2.0], "f2": [1.0, 2.0], "label": [0.0, None]})
    upload_and_validate_dataset(api_client, csv_bytes(bad))
    assert upload_and_validate_model(api_client, classifier_bytes)["compatibility"] == []


def test_deleting_a_model_keeps_past_evaluations_readable(api_client, classifier_bytes, classification_csv):
    from test_api_workflow import run_evaluation
    model = upload_and_validate_model(api_client, classifier_bytes)
    dataset = upload_and_validate_dataset(api_client, classification_csv)
    evaluation = run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()

    assert api_client.delete(f"/api/models/{model['id']}").status_code == 200
    assert api_client.delete(f"/api/datasets/{dataset['id']}").status_code == 200

    reopened = api_client.get(f"/api/evaluations/{evaluation['id']}")
    assert reopened.status_code == 200
    assert reopened.get_json()["model_filename"] == "model.pkl"
    assert api_client.get(f"/api/evaluations/{evaluation['id']}/charts/confusion_matrix.png").status_code == 200
