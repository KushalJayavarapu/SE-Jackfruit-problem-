"""Charts (FR-05, UC-08), reopening (UC-07, UC-10), comparison (FR-06, UC-09), reports (FR-07, UC-11)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

import config
from conftest import csv_bytes, pickle_bytes
from services import db, visualization_service
from test_api_workflow import run_evaluation, upload_and_validate_dataset, upload_and_validate_model

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def classification_eval(api_client, classifier_bytes, classification_csv):
    model = upload_and_validate_model(api_client, classifier_bytes)
    dataset = upload_and_validate_dataset(api_client, classification_csv)
    return run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()


@pytest.fixture
def regression_eval(api_client, regressor_bytes, regression_csv):
    model = upload_and_validate_model(api_client, regressor_bytes)
    dataset = upload_and_validate_dataset(api_client, regression_csv)
    return run_evaluation(api_client, model["id"], dataset["id"], "regression").get_json()


def owner_of(evaluation):
    return db.fetch_one("evaluations", "id = ?", (evaluation["id"],))["owner_id"]


def chart_kinds(evaluation):
    return [c["kind"] for c in evaluation["charts"]]


# ---------------- 08: charts depend on the task ----------------

def test_08_UT_01_classification_gets_confusion_matrix_and_roc_curve(classification_eval):
    assert chart_kinds(classification_eval) == ["confusion_matrix", "roc_curve"]


def test_08_UT_03_regression_gets_residual_and_predicted_vs_actual_plots(regression_eval):
    assert chart_kinds(regression_eval) == ["residuals", "predicted_vs_actual"]


def test_08_IT_03_no_classification_chart_for_regression_and_vice_versa(classification_eval, regression_eval):
    assert not {"residuals", "predicted_vs_actual"} & set(chart_kinds(classification_eval))
    assert not {"confusion_matrix", "roc_curve"} & set(chart_kinds(regression_eval))


def test_08_ST_02_no_roc_curve_when_the_model_gives_no_probabilities(api_client):
    X = pd.DataFrame({"x": [0.0, 1, 2, 3, 4, 5, 6, 7]})
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    model = upload_and_validate_model(api_client, pickle_bytes(LinearSVC().fit(X, y)))
    dataset = upload_and_validate_dataset(api_client, csv_bytes(X.assign(label=y)))

    result = run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()

    assert chart_kinds(result) == ["confusion_matrix"]
    assert result["metrics"]["roc_auc"] is None
    assert api_client.get(f"/api/evaluations/{result['id']}/charts/roc_curve.png").status_code == 404


def test_08_UT_02_each_available_chart_renders_a_png(api_client, classification_eval, regression_eval):
    for evaluation in (classification_eval, regression_eval):
        for chart in evaluation["charts"]:
            response = api_client.get(f"/api/evaluations/{evaluation['id']}/charts/{chart['kind']}.png")
            assert response.status_code == 200
            assert response.mimetype == "image/png"
            assert response.data.startswith(PNG) and len(response.data) > 3000


def test_unknown_chart_kind_is_404(api_client, classification_eval):
    response = api_client.get(f"/api/evaluations/{classification_eval['id']}/charts/pie.png")
    assert response.status_code == 404


def test_confusion_matrix_counts_match_the_predictions(constant_classifier, add_valid_model, add_valid_dataset):
    # Constant model predicts 1. True labels [1,1,0,1] -> TP=3, FP=1, TN=0, FN=0.
    from services import evaluation_service
    from conftest import owner
    model = add_valid_model(constant_classifier())
    dataset = add_valid_dataset(csv_bytes(pd.DataFrame({"x": [0.0, 1, 2, 3], "label": [1, 1, 0, 1]})))
    evaluation_service.evaluate_model(model["id"], dataset["id"], "classification", owner())

    saved = db.fetch_all("evaluations")[0]["charts"]["confusion_matrix"]
    assert saved["labels"] == ["0", "1"]
    assert saved["matrix"] == [[0, 1], [0, 3]]        # rows = actual, columns = predicted


def test_roc_curve_data_is_a_valid_curve(classification_eval):
    chart = db.fetch_all("evaluations")[0]["charts"]["roc_curve"]["curves"][0]
    assert chart["fpr"][0] == 0 and chart["tpr"][0] == 0
    assert chart["fpr"][-1] == 1 and chart["tpr"][-1] == 1
    assert chart["auc"] == pytest.approx(1.0)
    assert chart["fpr"] == sorted(chart["fpr"])


def test_multiclass_evaluation_gets_one_roc_curve_per_class(api_client):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=90), "b": rng.normal(size=90)})
    y = np.repeat([0, 1, 2], 30)
    X["a"] += y * 3
    model = upload_and_validate_model(api_client, pickle_bytes(LogisticRegression(max_iter=500).fit(X, y)))
    dataset = upload_and_validate_dataset(api_client, csv_bytes(X.assign(label=y)))

    result = run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()

    saved = db.fetch_all("evaluations")[0]["charts"]
    assert len(saved["roc_curve"]["curves"]) == 3
    assert len(saved["confusion_matrix"]["labels"]) == 3
    assert api_client.get(f"/api/evaluations/{result['id']}/charts/roc_curve.png").status_code == 200


def test_large_regression_charts_use_a_sample_but_metrics_use_every_row(api_client, monkeypatch, regressor_bytes):
    monkeypatch.setattr(config, "MAX_RESIDUAL_POINTS", 20)
    model = upload_and_validate_model(api_client, regressor_bytes)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=100), "f2": rng.normal(size=100)})
    dataset = upload_and_validate_dataset(api_client, csv_bytes(X.assign(label=2 * X.f1 + X.f2)))

    result = run_evaluation(api_client, model["id"], dataset["id"], "regression").get_json()

    points = db.fetch_all("evaluations")[0]["charts"]["residuals"]
    assert len(points["actual"]) == 20 and points["total_points"] == 100
    assert result["row_count"] == 100
    assert any("random sample of 20" in note for note in result["notes"])


# ---------------- 07 / 10: reopen saved evaluations ----------------

def test_07_IT_01_reopening_gives_exactly_the_stored_values(api_client, classification_eval):
    reopened = api_client.get(f"/api/evaluations/{classification_eval['id']}").get_json()
    assert reopened == classification_eval


def test_07_UT_02_missing_evaluation_gives_a_message_not_a_crash(api_client):
    api_response = api_client.get("/api/evaluations/does-not-exist")
    page_response = api_client.get("/evaluations/does-not-exist")
    assert api_response.status_code == 404 and "not available" in api_response.get_json()["error"]
    assert page_response.status_code == 404 and "not available" in page_response.get_data(as_text=True)


def test_07_UT_01_evaluation_page_shows_metrics_and_charts(api_client, classification_eval):
    page = api_client.get(f"/evaluations/{classification_eval['id']}").get_data(as_text=True)
    assert "Accuracy" in page and "ROC-AUC" in page
    assert "charts/confusion_matrix.png" in page and "charts/roc_curve.png" in page


def test_regression_evaluation_page_shows_only_regression_metrics(api_client, regression_eval):
    page = api_client.get(f"/evaluations/{regression_eval['id']}").get_data(as_text=True)
    assert "RMSE" in page and "R²" in page
    assert "ROC-AUC" not in page and "Accuracy" not in page       # DR-04


def test_10_UT_03_history_starts_empty(api_client):
    assert api_client.get("/api/evaluations").get_json() == []


def test_10_UT_02_history_lists_newest_first(api_client, classifier_bytes, classification_csv):
    model = upload_and_validate_model(api_client, classifier_bytes)
    dataset = upload_and_validate_dataset(api_client, classification_csv)
    first = run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()
    second = run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()
    ids = [e["id"] for e in api_client.get("/api/evaluations").get_json()]
    assert ids == [second["id"], first["id"]]


# ---------------- 09: compare ----------------

@pytest.fixture
def two_classification_evals(api_client, classifier_bytes, classification_csv):
    dataset = upload_and_validate_dataset(api_client, classification_csv)
    ids = []
    for _ in range(2):
        model = upload_and_validate_model(api_client, classifier_bytes)
        ids.append(run_evaluation(api_client, model["id"], dataset["id"], "classification").get_json()["id"])
    return ids


def test_09_IT_02_same_task_evaluations_are_compared_side_by_side(api_client, two_classification_evals):
    result = api_client.get("/api/compare", query_string={"ids": ",".join(two_classification_evals)})
    body = result.get_json()
    assert result.status_code == 200
    assert body["task_type"] == "classification"
    assert body["metric_keys"] == ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    assert [e["id"] for e in body["evaluations"]] == two_classification_evals       # order kept


def test_compare_page_and_chart_render(api_client, two_classification_evals):
    ids = ",".join(two_classification_evals)
    page = api_client.get(f"/compare?ids={ids}")
    assert page.status_code == 200 and "Comparing 2 classification evaluations" in page.get_data(as_text=True)
    chart = api_client.get(f"/api/compare/chart.png?ids={ids}")
    assert chart.status_code == 200 and chart.data.startswith(PNG)


def test_09_IT_01_different_task_types_are_not_compared_and_the_user_is_warned(api_client, classification_eval, regression_eval):
    ids = f"{classification_eval['id']},{regression_eval['id']}"
    response = api_client.get("/api/compare", query_string={"ids": ids})
    assert response.status_code == 400
    assert "different task types" in response.get_json()["error"]
    page = api_client.get(f"/compare?ids={ids}").get_data(as_text=True)
    assert "different task types" in page and "<table" not in page


def test_comparing_fewer_than_two_evaluations_is_refused(api_client, classification_eval):
    response = api_client.get("/api/compare", query_string={"ids": classification_eval["id"]})
    assert response.status_code == 400 and "two or more" in response.get_json()["error"]


def test_comparing_the_same_evaluation_twice_counts_as_one(api_client, classification_eval):
    ids = f"{classification_eval['id']},{classification_eval['id']}"
    assert api_client.get("/api/compare", query_string={"ids": ids}).status_code == 400


def test_comparing_with_a_missing_evaluation_is_refused(api_client, classification_eval):
    response = api_client.get("/api/compare", query_string={"ids": f"{classification_eval['id']},gone"})
    assert response.status_code == 404


def test_comparison_never_ranks_the_models(api_client, two_classification_evals):
    body = str(api_client.get("/api/compare", query_string={"ids": ",".join(two_classification_evals)}).get_json())
    assert "winner" not in body and "best" not in body


# ---------------- 11: reports ----------------

def test_11_UT_01_single_evaluation_report_is_a_pdf(api_client, classification_eval):
    response = api_client.get(f"/api/reports?ids={classification_eval['id']}")
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF-")
    assert "attachment" in response.headers["Content-Disposition"] and ".pdf" in response.headers["Content-Disposition"]


def test_11_UT_02_multi_evaluation_report_covers_all_of_them(api_client, two_classification_evals, regression_eval):
    ids = ",".join(two_classification_evals + [regression_eval["id"]])
    response = api_client.get(f"/api/reports?ids={ids}")
    assert response.status_code == 200 and response.data.startswith(b"%PDF-")
    single = api_client.get(f"/api/reports?ids={two_classification_evals[0]}")
    assert len(response.data) > len(single.data)


def test_11_UT_03_no_selection_gives_a_message(api_client):
    response = api_client.get("/api/reports")
    assert response.status_code == 400 and "at least one" in response.get_json()["error"]


def test_11_ST_02_report_is_refused_when_an_evaluation_no_longer_exists(api_client, classification_eval):
    response = api_client.get(f"/api/reports?ids={classification_eval['id']},gone")
    assert response.status_code == 404 and "no longer exist" in response.get_json()["error"]


def test_11_IT_02_report_is_built_from_saved_data_not_recomputed(api_client, classification_eval, monkeypatch):
    # Delete the files the evaluation was computed from: the report must still work.
    for model in api_client.get("/api/models").get_json():
        api_client.delete(f"/api/models/{model['id']}")
    for dataset in api_client.get("/api/datasets").get_json():
        api_client.delete(f"/api/datasets/{dataset['id']}")

    def forbidden(*args, **kwargs):
        raise AssertionError("report must not run models")
    monkeypatch.setattr("joblib.load", forbidden)

    response = api_client.get(f"/api/reports?ids={classification_eval['id']}")
    assert response.status_code == 200 and response.data.startswith(b"%PDF-")


def test_report_embeds_every_chart_as_an_image(api_client, classification_eval):
    from services import evaluation_service, report_service
    records = evaluation_service.get_many([classification_eval["id"]], owner_of(classification_eval))
    pdf = report_service.build_pdf(records, {"name": "Test Analyst", "email": "analyst@example.com"})
    assert pdf.count(b"/Subtype /Image") >= len(classification_eval["charts"])
    assert len(pdf) > 10_000


def test_report_is_the_same_for_the_same_saved_evaluation(api_client, classification_eval):
    from services import evaluation_service
    first = evaluation_service.get_many([classification_eval["id"]], owner_of(classification_eval))
    second = evaluation_service.get_many([classification_eval["id"]], owner_of(classification_eval))
    assert first[0]["metrics"] == second[0]["metrics"] == classification_eval["metrics"]


# ---------------- NFR-05: new charts are a local change ----------------

def test_adding_a_chart_only_needs_a_drawer_and_a_title(monkeypatch):
    monkeypatch.setitem(visualization_service.DRAWERS, "demo", lambda data: visualization_service._figure())
    png = visualization_service.render("demo", {"demo": {}})
    assert png.startswith(PNG)
