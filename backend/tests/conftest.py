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
from werkzeug.security import generate_password_hash

# Make `import app`, `import config`, `import services` work from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module  # noqa: E402
import config  # noqa: E402
from services import admin_service, auth_service, dataset_service, model_service  # noqa: E402


# --------------------------------------------------------------------------
# Isolated storage: every test gets its own empty uploads folders and its own
# SQLite database, so tests never touch the real backend/storage directory.
# Services read settings as config.NAME at call time, so patching config is enough.
# --------------------------------------------------------------------------
_state = {"owner_id": None}


def owner():
    """Id of the default account that service-level tests act as."""
    return _state["owner_id"]


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    datasets_dir = tmp_path / "datasets"
    models_dir.mkdir()
    datasets_dir.mkdir()
    monkeypatch.setattr(config, "MODELS_DIR", str(models_dir))
    monkeypatch.setattr(config, "DATASETS_DIR", str(datasets_dir))
    monkeypatch.setattr(config, "DATABASE_PATH", str(tmp_path / "test.db"))
    # Real password hashing is deliberately slow; use a cheap setting in tests.
    monkeypatch.setattr(auth_service, "generate_password_hash",
                        lambda password: generate_password_hash(password, method="pbkdf2:sha256:1000"))

    user = auth_service.register("Test Analyst", "analyst@example.com", "password123", "password123")
    _state["owner_id"] = user["id"]
    return tmp_path


class Api:
    """A test client for one browser session. Adds the CSRF header to unsafe requests."""

    def __init__(self, flask_client):
        self.client = flask_client

    def _token(self):
        with self.client.session_transaction() as session:
            has_token = "csrf_token" in session
        if not has_token:
            self.client.get("/login")
        with self.client.session_transaction() as session:
            return session["csrf_token"]

    def _send(self, method, url, **kwargs):
        if method != "get":
            headers = dict(kwargs.pop("headers", {}) or {})
            headers.setdefault("X-CSRF-Token", self._token())
            kwargs["headers"] = headers
        return getattr(self.client, method)(url, **kwargs)

    def get(self, url, **kwargs): return self._send("get", url, **kwargs)
    def post(self, url, **kwargs): return self._send("post", url, **kwargs)
    def put(self, url, **kwargs): return self._send("put", url, **kwargs)
    def delete(self, url, **kwargs): return self._send("delete", url, **kwargs)

    def login(self, email, password):
        return self.post("/login", data={"email": email, "password": password, "csrf_token": self._token()})

    def logout(self):
        return self.post("/logout")


@pytest.fixture
def new_client():
    """Factory for separate browser sessions."""
    app_module.app.config.update(TESTING=True)

    def make():
        return Api(app_module.app.test_client())
    return make


@pytest.fixture
def anon_client(new_client):
    return new_client()


@pytest.fixture
def make_user(new_client):
    """Create an account and return (api_client_logged_in, user_record)."""
    def make(email, role="analyst", name=None, password="password123"):
        name = name or email.split("@")[0].title()
        if role == "admin":
            user = admin_service.create_account(None, name, email, password, "admin")
        else:
            user = auth_service.register(name, email, password, password)
        client = new_client()
        assert client.login(email, password).status_code == 302
        return client, user
    return make


@pytest.fixture
def api_client(make_user):
    """A logged-in Analyst (the common case for workflow tests)."""
    client, _ = make_user("workflow@example.com")
    return client


@pytest.fixture
def admin_client(make_user):
    client, user = make_user("boss@example.com", role="admin", name="Boss")
    client.user = user
    return client


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
        record = model_service.upload_model(upload(data, filename), owner())
        return model_service.validate_model(record["id"], owner())
    return _add


@pytest.fixture
def add_valid_dataset():
    def _add(data, target="label", filename="data.csv"):
        record = dataset_service.upload_dataset(upload(data, filename), target, owner())
        return dataset_service.validate_dataset(record["id"], owner())
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
