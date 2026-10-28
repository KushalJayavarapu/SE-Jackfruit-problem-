"""
Integration and system tests that drive the real Flask app over HTTP with real
files (nothing mocked): upload -> validate -> evaluate, plus the error handling
added in BL-07. Test IDs refer to the project test plan.
"""
from io import BytesIO

import app as app_module


def post_model(client, data, filename="model.pkl"):
    return client.post("/api/models/upload", data={"model_file": (BytesIO(data), filename)})


def post_dataset(client, data, target="label", filename="data.csv"):
    return client.post(
        "/api/datasets/upload",
        data={"dataset_file": (BytesIO(data), filename), "target_variable": target},
    )


def upload_and_validate_model(client, data):
    model_id = post_model(client, data).get_json()["id"]
    return client.post(f"/api/models/{model_id}/validate").get_json()


def upload_and_validate_dataset(client, data, target="label"):
    dataset_id = post_dataset(client, data, target).get_json()["id"]
    return client.post(f"/api/datasets/{dataset_id}/validate").get_json()


def run_evaluation(client, model_id, dataset_id, task_type):
    return client.post(
        "/api/evaluations",
        json={"model_id": model_id, "dataset_id": dataset_id, "task_type": task_type},
    )


# ---------------- system tests: complete workflows ----------------

def test_03_ST_01_04_ST_01_06_ST_01_classification_workflow_end_to_end(
    api_client, classifier_bytes, classification_csv
):
    # Upload a model: stored, not yet validated.
    uploaded = post_model(api_client, classifier_bytes)
    assert uploaded.status_code == 201
    assert uploaded.get_json()["validation_status"] == "not_validated"

    # Validate it: now usable, and it reports the inputs it expects.
    model = upload_and_validate_model(api_client, classifier_bytes)
    assert model["validation_status"] == "valid"
    assert model["required_inputs"]["feature_names_in"] == ["f1", "f2"]

    # Upload and validate a dataset with a target column.
    dataset_upload = post_dataset(api_client, classification_csv)
    assert dataset_upload.status_code == 201
    dataset = upload_and_validate_dataset(api_client, classification_csv)
    assert dataset["validation_status"] == "valid"

    # Evaluate and check the stored result.
    response = run_evaluation(api_client, model["id"], dataset["id"], "classification")
    assert response.status_code == 201
    result = response.get_json()
    assert result["metrics"]["accuracy"] == 1.0
    assert result["row_count"] == 60

    history = api_client.get("/api/evaluations").get_json()
    assert [e["id"] for e in history] == [result["id"]]


def test_regression_workflow_end_to_end(api_client, regressor_bytes, regression_csv):
    model = upload_and_validate_model(api_client, regressor_bytes)
    dataset = upload_and_validate_dataset(api_client, regression_csv)

    response = run_evaluation(api_client, model["id"], dataset["id"], "regression")

    assert response.status_code == 201
    metrics = response.get_json()["metrics"]
    assert metrics["r2"] > 0.999
    assert metrics["mae"] < 1e-6


def test_several_models_can_be_evaluated_against_one_dataset(
    api_client, classifier_bytes, classification_csv
):
    dataset = upload_and_validate_dataset(api_client, classification_csv)
    first = upload_and_validate_model(api_client, classifier_bytes)
    second = upload_and_validate_model(api_client, classifier_bytes)

    assert run_evaluation(api_client, first["id"], dataset["id"], "classification").status_code == 201
    assert run_evaluation(api_client, second["id"], dataset["id"], "classification").status_code == 201
    assert len(api_client.get("/api/evaluations").get_json()) == 2


def test_uploads_appear_in_list_endpoints_and_on_the_home_page(
    api_client, classifier_bytes, classification_csv
):
    post_model(api_client, classifier_bytes, "my_model.pkl")
    post_dataset(api_client, classification_csv, filename="my_data.csv")

    assert [m["original_filename"] for m in api_client.get("/api/models").get_json()] == ["my_model.pkl"]
    assert [d["original_filename"] for d in api_client.get("/api/datasets").get_json()] == ["my_data.csv"]
    page = api_client.get("/").get_data(as_text=True)
    assert "my_model.pkl" in page and "my_data.csv" in page


def test_nothing_is_stored_for_rejected_uploads(api_client):
    post_model(api_client, b"data", "model.txt")
    post_dataset(api_client, b"a,b\n1,2\n3,4,5,6,7\n", target="a")

    assert api_client.get("/api/models").get_json() == []
    assert api_client.get("/api/datasets").get_json() == []


# ---------------- rejection paths over HTTP ----------------

def test_04_IT_03_invalid_model_file_is_rejected_immediately_with_a_reason(api_client):
    response = post_model(api_client, b"hello", "notes.txt")

    assert response.status_code == 400
    assert "Unsupported model file type" in response.get_json()["error"]


def test_04_UT_02_model_upload_without_a_file_is_rejected(api_client):
    response = api_client.post("/api/models/upload", data={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "No file was selected."


def test_03_ST_02_dataset_upload_without_a_target_is_rejected(api_client, classification_csv):
    response = post_dataset(api_client, classification_csv, target="")
    assert response.status_code == 400
    assert "target variable" in response.get_json()["error"]


def test_03_IT_02_corrupt_dataset_is_rejected_with_a_reason(api_client):
    response = post_dataset(api_client, b"a,b\n1,2\n3,4,5,6,7\n", target="a")
    assert response.status_code == 400
    assert "Could not read the dataset file" in response.get_json()["error"]


def test_corrupt_model_uploads_but_validation_flags_it(api_client):
    model = upload_and_validate_model(api_client, b"not a pickle")
    assert model["validation_status"] == "invalid"


def test_06_ST_02_evaluating_an_unvalidated_model_is_rejected_and_stores_nothing(
    api_client, classifier_bytes, classification_csv
):
    model_id = post_model(api_client, classifier_bytes).get_json()["id"]   # not validated
    dataset = upload_and_validate_dataset(api_client, classification_csv)

    response = run_evaluation(api_client, model_id, dataset["id"], "classification")

    assert response.status_code == 400
    assert "not passed validation" in response.get_json()["error"]
    assert api_client.get("/api/evaluations").get_json() == []


def test_05_ST_02_mismatched_task_type_is_rejected_over_http(
    api_client, classifier_bytes, classification_csv
):
    model = upload_and_validate_model(api_client, classifier_bytes)
    dataset = upload_and_validate_dataset(api_client, classification_csv)

    response = run_evaluation(api_client, model["id"], dataset["id"], "regression")

    assert response.status_code == 400
    assert "looks like a classifier" in response.get_json()["error"]


def test_validating_unknown_ids_returns_404_json(api_client):
    model_response = api_client.post("/api/models/missing/validate")
    dataset_response = api_client.post("/api/datasets/missing/validate")

    assert model_response.status_code == 404
    assert model_response.get_json() == {"error": "No such model upload."}
    assert dataset_response.status_code == 404
    assert dataset_response.get_json() == {"error": "No such dataset upload."}


# ---------------- BL-07: request validation and global error handling ----------------

def test_evaluation_request_must_be_json(api_client):
    response = api_client.post("/api/evaluations", data="model_id=1")
    assert response.status_code == 400
    assert "as JSON" in response.get_json()["error"]


def test_malformed_json_body_is_rejected(api_client):
    response = api_client.post(
        "/api/evaluations", data="{not valid json", content_type="application/json"
    )
    assert response.status_code == 400
    assert "valid JSON object" in response.get_json()["error"]


def test_json_body_that_is_not_an_object_is_rejected(api_client):
    response = api_client.post("/api/evaluations", json=["model_id", "dataset_id"])
    assert response.status_code == 400
    assert "valid JSON object" in response.get_json()["error"]


def test_missing_evaluation_fields_are_named_in_the_error(api_client):
    response = api_client.post("/api/evaluations", json={"model_id": "abc"})

    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "dataset_id" in error and "task_type" in error
    assert "model_id" not in error


def test_blank_and_non_string_evaluation_fields_are_rejected(api_client):
    response = api_client.post(
        "/api/evaluations", json={"model_id": "   ", "dataset_id": 123, "task_type": None}
    )
    assert response.status_code == 400
    error = response.get_json()["error"]
    assert "model_id" in error and "dataset_id" in error and "task_type" in error


def test_evaluation_with_unknown_ids_is_a_clean_400(api_client):
    response = run_evaluation(api_client, "nope", "nada", "classification")
    assert response.status_code == 400
    assert response.get_json() == {"error": "No such model."}


def test_oversized_upload_returns_413_json(api_client, monkeypatch, classifier_bytes):
    monkeypatch.setitem(app_module.app.config, "MAX_CONTENT_LENGTH", 100)
    response = post_model(api_client, classifier_bytes)

    assert response.status_code == 413
    assert "too large" in response.get_json()["error"]


def test_unknown_route_returns_json_404(api_client):
    response = api_client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.is_json and "error" in response.get_json()


def test_wrong_http_method_returns_json_405(api_client):
    response = api_client.get("/api/evaluations/run")        # no such route
    assert response.status_code == 404
    response = api_client.put("/api/models")                  # route exists, PUT not allowed
    assert response.status_code == 405
    assert response.is_json and "error" in response.get_json()


def test_unexpected_server_error_returns_generic_500_without_leaking_details(api_client, monkeypatch):
    def explode():
        raise RuntimeError("secret internal detail /etc/passwd")

    monkeypatch.setattr(app_module.model_service, "list_models", explode)
    response = api_client.get("/api/models")

    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Something went wrong" in body
    assert "secret internal detail" not in body
    assert "Traceback" not in body
