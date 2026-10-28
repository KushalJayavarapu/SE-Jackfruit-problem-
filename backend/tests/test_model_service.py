"""Unit tests for model upload (BL-01) and model validation (BL-02) — UC-04."""
import os

import pytest

from conftest import upload
from services import model_service
from services.model_service import ValidationError


# ---------------- BL-01: upload ----------------

def test_04_UT_01_supported_model_formats_are_accepted(classifier_bytes):
    for filename in ("model.pkl", "model.joblib", "MODEL.PKL"):
        record = model_service.upload_model(upload(classifier_bytes, filename))
        assert record["original_filename"] == filename
        assert os.path.exists(
            os.path.join(model_service.MODELS_DIR, record["stored_filename"])
        )


@pytest.mark.parametrize("filename", ["model.txt", "model.zip", "model", "model.pkl.exe"])
def test_04_UT_01_unsupported_model_formats_are_rejected(classifier_bytes, filename):
    with pytest.raises(ValidationError, match="Unsupported model file type"):
        model_service.upload_model(upload(classifier_bytes, filename))
    assert model_service.list_models() == []


def test_04_UT_02_no_file_selected_is_rejected():
    with pytest.raises(ValidationError, match="No file was selected"):
        model_service.upload_model(None)
    with pytest.raises(ValidationError, match="No file was selected"):
        model_service.upload_model(upload(b"data", ""))


def test_empty_model_file_is_rejected():
    with pytest.raises(ValidationError, match="empty"):
        model_service.upload_model(upload(b"", "model.pkl"))


def test_model_over_size_limit_is_rejected(classifier_bytes, monkeypatch):
    monkeypatch.setattr(model_service, "MAX_MODEL_SIZE_BYTES", 100)
    assert len(classifier_bytes) > 100
    with pytest.raises(ValidationError, match="exceeds the"):
        model_service.upload_model(upload(classifier_bytes, "model.pkl"))
    assert model_service.list_models() == []


def test_model_exactly_at_size_limit_is_accepted(classifier_bytes, monkeypatch):
    monkeypatch.setattr(model_service, "MAX_MODEL_SIZE_BYTES", len(classifier_bytes))
    record = model_service.upload_model(upload(classifier_bytes, "model.pkl"))
    assert record["size_bytes"] == len(classifier_bytes)


def test_04_UT_03_uploaded_model_becomes_available(classifier_bytes):
    record = model_service.upload_model(upload(classifier_bytes, "model.pkl"))

    assert record["validation_status"] == "not_validated"
    assert record["size_bytes"] == len(classifier_bytes)
    assert model_service.get_model(record["id"]) == record
    assert [m["id"] for m in model_service.list_models()] == [record["id"]]


def test_each_upload_gets_a_unique_id_and_stored_file(classifier_bytes):
    first = model_service.upload_model(upload(classifier_bytes, "same.pkl"))
    second = model_service.upload_model(upload(classifier_bytes, "same.pkl"))
    assert first["id"] != second["id"]
    assert first["stored_filename"] != second["stored_filename"]


# ---------------- BL-02: validate ----------------

def test_valid_classifier_passes_validation_and_reports_inputs(classifier_bytes):
    record = model_service.upload_model(upload(classifier_bytes, "model.pkl"))
    result = model_service.validate_model(record["id"])

    assert result["validation_status"] == "valid"
    assert result["required_inputs"]["model_type"] == "LogisticRegression"
    assert result["required_inputs"]["n_features_in"] == 2
    assert result["required_inputs"]["feature_names_in"] == ["f1", "f2"]


def test_valid_regressor_passes_validation(regressor_bytes):
    record = model_service.upload_model(upload(regressor_bytes, "model.pkl"))
    result = model_service.validate_model(record["id"])
    assert result["validation_status"] == "valid"
    assert result["required_inputs"]["model_type"] == "LinearRegression"


def test_validation_result_is_persisted(classifier_bytes):
    record = model_service.upload_model(upload(classifier_bytes, "model.pkl"))
    model_service.validate_model(record["id"])
    assert model_service.get_model(record["id"])["validation_status"] == "valid"


def test_corrupt_model_file_is_marked_invalid():
    record = model_service.upload_model(upload(b"this is not a pickle", "model.pkl"))
    result = model_service.validate_model(record["id"])

    assert result["validation_status"] == "invalid"
    assert "Could not load the model file" in result["validation_message"]
    assert result["required_inputs"] is None


def test_loadable_object_without_predict_is_marked_invalid():
    import joblib
    from io import BytesIO

    buffer = BytesIO()
    joblib.dump({"not": "a model"}, buffer)
    record = model_service.upload_model(upload(buffer.getvalue(), "model.pkl"))
    result = model_service.validate_model(record["id"])

    assert result["validation_status"] == "invalid"
    assert "predict()" in result["validation_message"]


def test_validating_unknown_model_id_raises():
    with pytest.raises(ValidationError, match="No such model"):
        model_service.validate_model("does-not-exist")
