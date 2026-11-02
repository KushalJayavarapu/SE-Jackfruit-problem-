"""Smoke tests: pages render for a signed-in user and static files are served."""
import pytest

PAGES = ["/dashboard", "/models", "/datasets", "/evaluate", "/evaluations"]


@pytest.mark.parametrize("path", PAGES)
def test_workspace_pages_render_for_a_signed_in_analyst(api_client, path):
    response = api_client.get(path)
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert 'href="#main-content"' in page          # skip link
    assert 'name="csrf-token"' in page


def test_page_forms_have_the_ids_the_script_expects(api_client):
    assert 'id="model-upload-form"' in api_client.get("/models").get_data(as_text=True)
    assert 'id="dataset-upload-form"' in api_client.get("/datasets").get_data(as_text=True)
    assert 'id="evaluate-form"' in api_client.get("/evaluate").get_data(as_text=True)
    assert 'id="evaluation-history"' in api_client.get("/evaluations").get_data(as_text=True)


def test_static_assets_are_served(anon_client):
    assert anon_client.get("/static/app.js").status_code == 200
    assert anon_client.get("/static/style.css").status_code == 200


def test_login_and_register_pages_render_without_an_account(anon_client):
    for path in ("/login", "/register"):
        response = anon_client.get(path)
        assert response.status_code == 200
        assert "Model Evaluator" in response.get_data(as_text=True)
