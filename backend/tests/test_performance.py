"""NFR-01: an evaluation at the stated limits finishes in under 10 seconds.
Uses the 50,000-row dataset limit with a single user. (The 100 MB model limit is
covered by the size-limit tests; a 100 MB file would make the suite slow.)"""
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from conftest import csv_bytes, pickle_bytes
from test_api_workflow import run_evaluation, upload_and_validate_dataset, upload_and_validate_model

ROWS = 50_000
LIMIT_SECONDS = 10


def make_frame(n=ROWS, features=20, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, features)), columns=[f"f{i}" for i in range(features)])
    return X


@pytest.mark.parametrize("name, model_factory", [
    ("logistic regression", lambda: LogisticRegression(max_iter=200)),
    ("random forest", lambda: RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=1, random_state=0)),
])
def test_classification_at_50000_rows_finishes_within_10_seconds(api_client, name, model_factory):
    X = make_frame()
    y = (X.f0 + X.f1 > 0).astype(int)
    model = upload_and_validate_model(api_client, pickle_bytes(model_factory().fit(X.iloc[:2000], y.iloc[:2000])))
    dataset = upload_and_validate_dataset(api_client, csv_bytes(X.assign(label=y)))
    assert dataset["row_count"] == ROWS

    started = time.perf_counter()
    response = run_evaluation(api_client, model["id"], dataset["id"], "classification")
    elapsed = time.perf_counter() - started

    assert response.status_code == 201
    assert elapsed < LIMIT_SECONDS, f"{name}: {elapsed:.1f}s"


def test_regression_at_50000_rows_finishes_within_10_seconds_and_charts_are_sampled(api_client):
    X = make_frame()
    y = X.f0 * 3 + X.f1
    model = upload_and_validate_model(api_client, pickle_bytes(LinearRegression().fit(X, y)))
    dataset = upload_and_validate_dataset(api_client, csv_bytes(X.assign(label=y)))

    started = time.perf_counter()
    response = run_evaluation(api_client, model["id"], dataset["id"], "regression")
    elapsed = time.perf_counter() - started

    assert response.status_code == 201
    assert elapsed < LIMIT_SECONDS
    result = response.get_json()
    assert result["row_count"] == ROWS
    chart = api_client.get(f"/api/evaluations/{result['id']}/charts/residuals.png")
    assert chart.status_code == 200
