"""
Shared pytest fixtures for the backend tests.

Test IDs in the test names (e.g. 04_UT_01) refer to the cases in
documents/Web_Based_ML_Model_Evaluator_Test_Cases.pdf:
  03_* Upload Dataset   04_* Upload Model
  05_* Select Task Type 06_* Evaluate Model
  UT = unit, IT = integration, ST = system (end to end)
"""
import sys
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from werkzeug.datastructures import FileStorage

# Make `import app`, `import config`, `import services` work from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
import config  # noqa: E402
from services import dataset_service, evaluation_service, model_service  # noqa: E402


# --------------------------------------------------------------------------
# Isolated storage: every test gets its own empty uploads folder and indexes,
# so tests never read or write the real backend/storage directory.
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    datasets_dir = tmp_path / "datasets"
    models_dir.mkdir()
    datasets_dir.mkdir()

    paths = {
        "MODELS_DIR": str(models_dir),
        "DATASETS_DIR": str(datasets_dir),
        "MODELS_INDEX": str(tmp_path / "models.json"),
        "DATASETS_INDEX": str(tmp_path / "datasets.json"),
        "EVALUATIONS_INDEX": str(tmp_path / "evaluations.json"),
    }
    # The services do `from config import X`, so each module holds its own
    # reference to these values and every one of them has to be patched.
    for module in (config, model_service, dataset_service, evaluation_service):
        for name, value in paths.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)
    return tmp_path


@pytest.fixture
def api_client():
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as client:
        yield client


# --------------------------------------------------------------------------
# Builders for real model / dataset files
# --------------------------------------------------------------------------
def pickle_bytes(model):
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()


def csv_bytes(frame):
    return frame.to_csv(index=False).encode()


def upload(data, filename):
    """Wrap raw bytes the way Flask hands an uploaded file to a service."""
    return FileStorage(stream=BytesIO(data), filename=filename)


def _features(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})


def _classification_data():
    """Two clearly separated classes, so a fitted model scores 1.0 exactly."""
    X = _features()
    y = (X.f1 > 0).astype(int)
    X["f1"] = X["f1"] + np.where(y == 1, 3.0, -3.0)
    return X, y


@pytest.fixture
def classifier_bytes():
    X, y = _classification_data()
    return pickle_bytes(LogisticRegression().fit(X, y))


@pytest.fixture
def regressor_bytes():
    X = _features()
    y = 2 * X.f1 + X.f2
    return pickle_bytes(LinearRegression().fit(X, y))


@pytest.fixture
def classification_csv():
    X, y = _classification_data()
    return csv_bytes(X.assign(label=y))


@pytest.fixture
def regression_csv():
    X = _features()
    return csv_bytes(X.assign(label=2 * X.f1 + X.f2))


# --------------------------------------------------------------------------
# Helpers that put records straight into storage through the real services
# --------------------------------------------------------------------------
@pytest.fixture
def add_valid_model():
    def _add(data, filename="model.pkl"):
        record = model_service.upload_model(upload(data, filename))
        return model_service.validate_model(record["id"])
    return _add


@pytest.fixture
def add_valid_dataset():
    def _add(data, target="label", filename="data.csv"):
        record = dataset_service.upload_dataset(upload(data, filename), target)
        return dataset_service.validate_dataset(record["id"])
    return _add


@pytest.fixture
def constant_classifier():
    """Always predicts class 1, so metrics can be worked out by hand."""
    def _make(feature_names=("x",)):
        X = pd.DataFrame({name: [0.0, 1.0, 2.0, 3.0] for name in feature_names})
        return pickle_bytes(
            DummyClassifier(strategy="constant", constant=1).fit(X, [0, 1, 1, 1])
        )
    return _make


@pytest.fixture
def constant_regressor():
    """Always predicts 2.0, so metrics can be worked out by hand."""
    def _make():
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        return pickle_bytes(DummyRegressor(strategy="constant", constant=2.0).fit(X, [1, 2, 3, 4]))
    return _make
