"""NFR-02: one Analyst can never reach another Analyst's models, datasets or evaluations,
even by guessing an id. Admins manage accounts, not other people's data."""
import pytest

from test_api_workflow import post_dataset, post_model, run_evaluation, upload_and_validate_dataset, upload_and_validate_model


@pytest.fixture
def two_analysts(make_user):
    alice, _ = make_user("alice@example.com")
    bob, _ = make_user("bob@example.com")
    return alice, bob


@pytest.fixture
def alice_data(two_analysts, classifier_bytes, classification_csv):
    alice, bob = two_analysts
    model = upload_and_validate_model(alice, classifier_bytes)
    dataset = upload_and_validate_dataset(alice, classification_csv)
    evaluation = run_evaluation(alice, model["id"], dataset["id"], "classification").get_json()
    return {"model": model, "dataset": dataset, "evaluation": evaluation}


def test_lists_only_show_your_own_records(two_analysts, alice_data):
    alice, bob = two_analysts
    assert len(alice.get("/api/models").get_json()) == 1
    assert bob.get("/api/models").get_json() == []
    assert bob.get("/api/datasets").get_json() == []
    assert bob.get("/api/evaluations").get_json() == []


def test_bob_cannot_validate_alices_model_or_dataset_by_id(two_analysts, alice_data):
    _, bob = two_analysts
    assert bob.post(f"/api/models/{alice_data['model']['id']}/validate").status_code == 404
    assert bob.post(f"/api/datasets/{alice_data['dataset']['id']}/validate").status_code == 404


def test_bob_cannot_delete_alices_files(two_analysts, alice_data):
    alice, bob = two_analysts
    assert bob.delete(f"/api/models/{alice_data['model']['id']}").status_code == 404
    assert bob.delete(f"/api/datasets/{alice_data['dataset']['id']}").status_code == 404
    assert len(alice.get("/api/models").get_json()) == 1


def test_bob_cannot_evaluate_with_alices_files(two_analysts, alice_data, classifier_bytes, classification_csv):
    _, bob = two_analysts
    own_model = upload_and_validate_model(bob, classifier_bytes)
    own_dataset = upload_and_validate_dataset(bob, classification_csv)

    stolen_model = run_evaluation(bob, alice_data["model"]["id"], own_dataset["id"], "classification")
    stolen_dataset = run_evaluation(bob, own_model["id"], alice_data["dataset"]["id"], "classification")

    assert stolen_model.status_code == 400 and stolen_model.get_json()["error"] == "No such model."
    assert stolen_dataset.status_code == 400 and stolen_dataset.get_json()["error"] == "No such dataset."


def test_bob_cannot_open_alices_evaluation_by_id(two_analysts, alice_data):
    alice, bob = two_analysts
    eid = alice_data["evaluation"]["id"]
    assert alice.get(f"/api/evaluations/{eid}").status_code == 200
    assert bob.get(f"/api/evaluations/{eid}").status_code == 404
    page = bob.get(f"/evaluations/{eid}")
    assert page.status_code == 404
    assert "model.pkl" not in page.get_data(as_text=True)          # nothing from the evaluation leaks


def test_bob_cannot_fetch_alices_charts_comparison_or_report(two_analysts, alice_data):
    _, bob = two_analysts
    eid = alice_data["evaluation"]["id"]
    assert bob.get(f"/api/evaluations/{eid}/charts/confusion_matrix.png").status_code == 404
    assert bob.get(f"/api/reports?ids={eid}").status_code == 404
    assert bob.get(f"/api/compare?ids={eid},{eid}").status_code in (400, 404)


def test_a_report_cannot_smuggle_in_another_users_evaluation(two_analysts, alice_data, classifier_bytes, classification_csv):
    alice, bob = two_analysts
    own = run_evaluation(bob, upload_and_validate_model(bob, classifier_bytes)["id"],
                         upload_and_validate_dataset(bob, classification_csv)["id"], "classification").get_json()
    response = bob.get(f"/api/reports?ids={own['id']},{alice_data['evaluation']['id']}")
    assert response.status_code == 404


def test_uploaded_file_names_are_scoped_to_the_owner_in_listings(two_analysts, classifier_bytes):
    alice, bob = two_analysts
    post_model(alice, classifier_bytes, "alice_secret_model.pkl")
    assert "alice_secret_model" not in str(bob.get("/api/models").get_json())


def test_admin_cannot_read_an_analysts_data(admin_client, two_analysts, alice_data):
    assert admin_client.get("/api/models").get_json() == []
    assert admin_client.get(f"/api/evaluations/{alice_data['evaluation']['id']}").status_code == 404


def test_cross_user_dataset_compatibility_report_never_mentions_other_users_files(two_analysts, classifier_bytes, classification_csv):
    alice, bob = two_analysts
    post_dataset(alice, classification_csv, filename="alice_private.csv")
    upload_and_validate_dataset(alice, classification_csv)
    model = upload_and_validate_model(bob, classifier_bytes)
    assert model["compatibility"] == []
