"""Unit tests for dataset upload (BL-03) and dataset validation (BL-04) — UC-03."""
import os

import pandas as pd
import pytest

from conftest import csv_bytes, upload
from services import dataset_service
from services.model_service import ValidationError


def small_frame(**overrides):
    data = {"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "label": [0, 1, 0]}
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------- BL-03: upload ----------------

def test_03_UT_01_csv_dataset_is_accepted():
    record = dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "label")

    assert record["original_filename"] == "data.csv"
    assert record["row_count"] == 3
    assert record["columns"] == ["f1", "f2", "label"]
    assert record["target_variable"] == "label"
    assert record["validation_status"] == "not_validated"
    assert os.path.exists(os.path.join(dataset_service.DATASETS_DIR, record["stored_filename"]))


@pytest.mark.parametrize("filename", ["data.txt", "data.xlsx", "data.json", "data"])
def test_03_UT_01_unsupported_dataset_formats_are_rejected(filename):
    with pytest.raises(ValidationError, match="Unsupported dataset file type"):
        dataset_service.upload_dataset(upload(csv_bytes(small_frame()), filename), "label")
    assert dataset_service.list_datasets() == []


def test_no_file_selected_is_rejected():
    with pytest.raises(ValidationError, match="No file was selected"):
        dataset_service.upload_dataset(None, "label")


@pytest.mark.parametrize("bad_csv", [
    b"a,b\n1,2\n3,4,5,6,7\n",      # rows with the wrong number of fields
    b"",                            # nothing at all
    bytes(range(256)) * 4,          # binary garbage
])
def test_03_UT_02_corrupt_dataset_is_rejected(bad_csv):
    with pytest.raises(ValidationError, match="Could not read the dataset file"):
        dataset_service.upload_dataset(upload(bad_csv, "data.csv"), "a")
    assert dataset_service.list_datasets() == []


def test_header_only_dataset_with_no_rows_is_rejected():
    with pytest.raises(ValidationError, match="no rows"):
        dataset_service.upload_dataset(upload(b"a,b\n", "data.csv"), "a")


def test_03_UT_03_target_column_must_be_specified():
    for blank in ("", "   ", None):
        with pytest.raises(ValidationError, match="target variable must be specified"):
            dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), blank)


def test_03_UT_03_target_column_must_exist_and_error_lists_columns():
    with pytest.raises(ValidationError) as exc:
        dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "wrong")
    message = str(exc.value)
    assert "'wrong' is not a column" in message
    assert "f1" in message and "f2" in message and "label" in message


def test_target_name_is_trimmed():
    record = dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "  label  ")
    assert record["target_variable"] == "label"


def test_dataset_over_row_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(dataset_service, "MAX_DATASET_ROWS", 2)
    with pytest.raises(ValidationError, match="exceeding the 2 row limit"):
        dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "label")


def test_dataset_exactly_at_row_limit_is_accepted(monkeypatch):
    monkeypatch.setattr(dataset_service, "MAX_DATASET_ROWS", 3)
    record = dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "label")
    assert record["row_count"] == 3


def test_uploaded_dataset_is_listed_and_retrievable():
    record = dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "label")
    assert dataset_service.get_dataset(record["id"]) == record
    assert [d["id"] for d in dataset_service.list_datasets()] == [record["id"]]


# ---------------- BL-04: validate ----------------

def test_clean_dataset_passes_validation_with_schema_summary():
    record = dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "label")
    result = dataset_service.validate_dataset(record["id"])

    assert result["validation_status"] == "valid"
    summary = result["schema_summary"]
    assert summary["row_count"] == 3
    assert summary["feature_columns"] == ["f1", "f2"]
    assert summary["target_variable"] == "label"
    assert summary["duplicate_rows"] == 0
    by_name = {c["name"]: c for c in summary["columns"]}
    assert by_name["label"]["is_target"] is True
    assert by_name["f1"]["is_target"] is False
    assert by_name["f1"]["missing_count"] == 0


def test_missing_values_in_features_are_reported_but_still_valid():
    frame = small_frame(f1=[1.0, None, 3.0])
    record = dataset_service.upload_dataset(upload(csv_bytes(frame), "data.csv"), "label")
    result = dataset_service.validate_dataset(record["id"])

    assert result["validation_status"] == "valid"
    by_name = {c["name"]: c for c in result["schema_summary"]["columns"]}
    assert by_name["f1"]["missing_count"] == 1


def test_missing_target_values_make_dataset_invalid():
    frame = small_frame(label=[0.0, None, 1.0])
    record = dataset_service.upload_dataset(upload(csv_bytes(frame), "data.csv"), "label")
    result = dataset_service.validate_dataset(record["id"])

    assert result["validation_status"] == "invalid"
    assert "1 missing value" in result["validation_message"]
    assert result["schema_summary"] is None


def test_dataset_with_only_a_target_column_is_invalid():
    frame = pd.DataFrame({"label": [0, 1, 0]})
    record = dataset_service.upload_dataset(upload(csv_bytes(frame), "data.csv"), "label")
    result = dataset_service.validate_dataset(record["id"])

    assert result["validation_status"] == "invalid"
    assert "no feature columns" in result["validation_message"]


def test_duplicate_rows_are_counted():
    frame = pd.DataFrame({"f1": [1, 1, 2], "label": [0, 0, 1]})
    record = dataset_service.upload_dataset(upload(csv_bytes(frame), "data.csv"), "label")
    result = dataset_service.validate_dataset(record["id"])
    assert result["schema_summary"]["duplicate_rows"] == 1


def test_validation_result_is_persisted():
    record = dataset_service.upload_dataset(upload(csv_bytes(small_frame()), "data.csv"), "label")
    dataset_service.validate_dataset(record["id"])
    assert dataset_service.get_dataset(record["id"])["validation_status"] == "valid"


def test_validating_unknown_dataset_id_raises():
    with pytest.raises(ValidationError, match="No such dataset"):
        dataset_service.validate_dataset("does-not-exist")
