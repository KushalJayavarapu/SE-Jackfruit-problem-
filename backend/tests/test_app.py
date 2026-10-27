"""Smoke tests for the Flask interface and its existing API routes."""
from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module


def test_home_page_renders_accessible_workflows(client):
    response = client.get("/")

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    for expected in (
        'id="model-upload-form"', 'id="dataset-upload-form"',
        'id="model-validate-form"', 'id="dataset-validate-form"',
        'id="evaluate-form"', 'id="evaluation-history"',
        'aria-live="polite"', 'href="#main-content"',
    ):
        assert expected in page


def test_static_assets_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_existing_list_routes_return_json_lists(client, monkeypatch):
    monkeypatch.setattr(app_module.model_service, "list_models", lambda: [{"id": "m1"}])
    monkeypatch.setattr(app_module.dataset_service, "list_datasets", lambda: [{"id": "d1"}])
    monkeypatch.setattr(app_module.evaluation_service, "list_evaluations", lambda: [{"id": "e1"}])

    assert client.get("/api/models").get_json() == [{"id": "m1"}]
    assert client.get("/api/datasets").get_json() == [{"id": "d1"}]
    assert client.get("/api/evaluations").get_json() == [{"id": "e1"}]


def test_upload_routes_keep_existing_validation_error_schema(client, monkeypatch):
    def reject(*args, **kwargs):
        raise app_module.ValidationError("A useful validation message.")

    monkeypatch.setattr(app_module.model_service, "upload_model", reject)
    monkeypatch.setattr(app_module.dataset_service, "upload_dataset", reject)

    model_response = client.post("/api/models/upload", data={"model_file": (BytesIO(b"model"), "model.pkl")})
    dataset_response = client.post("/api/datasets/upload", data={"dataset_file": (BytesIO(b"x,y\n1,2"), "data.csv"), "target_variable": "y"})

    assert model_response.status_code == 400
    assert model_response.get_json() == {"error": "A useful validation message."}
    assert dataset_response.status_code == 400
    assert dataset_response.get_json() == {"error": "A useful validation message."}


import pytest


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client
