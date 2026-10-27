
import sys
from pathlib import Path

import pytest

# Make the backend directory importable when tests run from the repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def test_evaluation_rejects_non_json_request(client):
    response = client.post("/api/evaluations", data="not json")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_evaluation_rejects_empty_json_object(client):
    response = client.post("/api/evaluations", json={})

    assert response.status_code == 400
    assert "model_id" in response.get_json()["error"]


def test_evaluation_rejects_blank_required_field(client):
    response = client.post(
        "/api/evaluations",
        json={
            "model_id": "model-1",
            "dataset_id": "dataset-1",
            "task_type": " ",
        },
    )

    assert response.status_code == 400
    assert "task_type" in response.get_json()["error"]


def test_oversized_upload_returns_json_error(client):
    app_module.app.config["MAX_CONTENT_LENGTH"] = 10

    try:
        response = client.post(
            "/api/models/upload",
            data={"model_file": ( __import__("io").BytesIO(b"x" * 100), "model.pkl")},
            content_type="multipart/form-data",
        )
    finally:
        app_module.app.config["MAX_CONTENT_LENGTH"] = 110 * 1024 * 1024

    assert response.status_code == 413
    assert "error" in response.get_json()
