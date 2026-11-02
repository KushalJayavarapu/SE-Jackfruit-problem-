"""Registration and login (FR-01, UC-01, UC-02) and session / CSRF behaviour."""
import pytest

from services import admin_service, auth_service, db

GOOD = {"name": "Ada Lovelace", "email": "ada@example.com", "password": "correct horse", "confirm_password": "correct horse"}


def register(client, **overrides):
    data = {**GOOD, **overrides}
    return client.post("/register", data={**data, "csrf_token": client._token()})


# ---------------- UC-01 registration ----------------

def test_01_IT_03_valid_registration_creates_an_analyst_and_signs_in(anon_client):
    response = register(anon_client)
    assert response.status_code == 302 and response.headers["Location"].endswith("/dashboard")

    me = anon_client.get("/api/me").get_json()
    assert me["email"] == "ada@example.com" and me["role"] == "analyst" and me["status"] == "active"
    assert "password_hash" not in me
    assert anon_client.get("/dashboard").status_code == 200


def test_01_UT_01_blank_required_fields_are_rejected(anon_client):
    response = register(anon_client, name="", email="", password="", confirm_password="")
    assert response.status_code == 400
    page = response.get_data(as_text=True)
    assert "Enter your name." in page and "Enter your email address." in page
    assert db.count("users", "email = ?", ("",)) == 0


@pytest.mark.parametrize("email", ["plain", "a@b", "@example.com", "a b@example.com"])
def test_01_UT_02_invalid_email_is_rejected(anon_client, email):
    response = register(anon_client, email=email)
    assert response.status_code == 400
    assert "valid email address" in response.get_data(as_text=True)


def test_01_UT_02_short_password_is_rejected(anon_client):
    response = register(anon_client, password="short", confirm_password="short")
    assert response.status_code == 400
    assert "at least 8 characters" in response.get_data(as_text=True)


def test_mismatched_password_confirmation_is_rejected(anon_client):
    response = register(anon_client, confirm_password="something else")
    assert response.status_code == 400
    assert "do not match" in response.get_data(as_text=True)


def test_01_UT_03_duplicate_email_is_rejected_ignoring_case(anon_client):
    assert register(anon_client).status_code == 302
    other = anon_client.__class__(anon_client.client.application.test_client())
    response = register(other, email="ADA@Example.com")
    assert response.status_code == 400
    assert "already exists" in response.get_data(as_text=True)


def test_01_IT_02_registration_cannot_create_an_admin(anon_client):
    # An attacker adds a role field to the form: it must be ignored.
    response = anon_client.post("/register", data={**GOOD, "role": "admin", "csrf_token": anon_client._token()})
    assert response.status_code == 302
    assert anon_client.get("/api/me").get_json()["role"] == "analyst"
    assert admin_service.list_users()[-1]["role"] == "analyst"


def test_passwords_are_stored_hashed(anon_client):
    register(anon_client)
    stored = db.fetch_one("users", "email = ?", ("ada@example.com",))["password_hash"]
    assert "correct horse" not in stored and stored != "correct horse"


def test_email_is_stored_lowercase(anon_client):
    register(anon_client, email="Ada@Example.COM")
    assert db.fetch_one("users", "email = ?", ("ada@example.com",)) is not None


# ---------------- UC-02 login ----------------

def test_02_IT_01_analyst_login_goes_to_the_analyst_dashboard(make_user, new_client):
    auth_service.register("Sam", "sam@example.com", "password123", "password123")
    client = new_client()
    response = client.login("sam@example.com", "password123")
    assert response.status_code == 302 and response.headers["Location"].endswith("/dashboard")
    assert client.get("/dashboard").status_code == 200


def test_02_IT_02_admin_login_goes_to_the_admin_dashboard(admin_client, new_client):
    client = new_client()
    response = client.login("boss@example.com", "password123")
    assert response.status_code == 302 and response.headers["Location"].endswith("/admin/users")


def test_02_UT_03_wrong_password_and_unknown_email_give_the_same_message(new_client, make_user):
    make_user("known@example.com")
    client = new_client()
    wrong = client.login("known@example.com", "not the password")
    unknown = client.login("nobody@example.com", "password123")
    assert wrong.status_code == unknown.status_code == 400
    assert "Incorrect email or password." in wrong.get_data(as_text=True)
    assert "Incorrect email or password." in unknown.get_data(as_text=True)
    assert client.get("/api/me").status_code == 401


def test_02_UT_01_blank_login_fields_are_rejected(new_client):
    response = new_client().login("", "")
    assert response.status_code == 400
    assert "Enter your email and password." in response.get_data(as_text=True)


def test_login_is_case_insensitive_for_email(new_client, make_user):
    make_user("case@example.com")
    assert new_client().login("CASE@Example.com", "password123").status_code == 302


def test_02_IT_03_suspended_account_cannot_log_in(admin_client, make_user, new_client):
    _, victim = make_user("victim@example.com")
    admin_client.post(f"/api/admin/users/{victim['id']}/status", json={"status": "suspended"})

    response = new_client().login("victim@example.com", "password123")

    assert response.status_code == 400
    assert "suspended" in response.get_data(as_text=True)


def test_suspending_an_account_ends_its_open_session_immediately(admin_client, make_user):
    victim_client, victim = make_user("live@example.com")
    assert victim_client.get("/api/models").status_code == 200

    admin_client.post(f"/api/admin/users/{victim['id']}/status", json={"status": "suspended"})

    assert victim_client.get("/api/models").status_code == 401
    assert victim_client.get("/dashboard").status_code == 302


def test_reactivated_account_can_log_in_again(admin_client, make_user, new_client):
    _, user = make_user("back@example.com")
    admin_client.post(f"/api/admin/users/{user['id']}/status", json={"status": "suspended"})
    admin_client.post(f"/api/admin/users/{user['id']}/status", json={"status": "active"})
    assert new_client().login("back@example.com", "password123").status_code == 302


def test_logout_ends_the_session(api_client):
    assert api_client.get("/api/models").status_code == 200
    assert api_client.logout().status_code == 302
    assert api_client.get("/api/models").status_code == 401


def test_last_login_time_is_recorded(make_user):
    _, user = make_user("stamp@example.com")
    assert db.fetch_one("users", "id = ?", (user["id"],))["last_login_at"].endswith("Z")


# ---------------- access control for anonymous visitors ----------------

@pytest.mark.parametrize("path", ["/dashboard", "/models", "/datasets", "/evaluate", "/evaluations", "/compare", "/admin/users"])
def test_pages_redirect_anonymous_visitors_to_login(anon_client, path):
    response = anon_client.get(path)
    assert response.status_code == 302 and "/login" in response.headers["Location"]


@pytest.mark.parametrize("path", ["/api/models", "/api/datasets", "/api/evaluations", "/api/me", "/api/admin/users", "/api/reports?ids=x"])
def test_api_returns_401_json_for_anonymous_visitors(anon_client, path):
    response = anon_client.get(path)
    assert response.status_code == 401
    assert response.get_json() == {"error": "Please log in to continue."}


def test_login_redirects_back_to_the_requested_page(new_client, make_user):
    make_user("next@example.com")
    client = new_client()
    client.get("/models")
    response = client.post("/login?next=/models", data={"email": "next@example.com", "password": "password123", "csrf_token": client._token()})
    assert response.headers["Location"].endswith("/models")


@pytest.mark.parametrize("target", ["https://evil.example/", "//evil.example/", "javascript:alert(1)"])
def test_login_does_not_redirect_to_other_sites(new_client, make_user, target):
    make_user("safe@example.com")
    client = new_client()
    response = client.post("/login", query_string={"next": target},
                           data={"email": "safe@example.com", "password": "password123", "csrf_token": client._token()})
    assert response.status_code == 302
    assert "evil" not in response.headers["Location"] and "javascript" not in response.headers["Location"]


# ---------------- CSRF ----------------

def test_post_without_csrf_token_is_rejected(api_client):
    response = api_client.client.post("/api/models/upload", data={})
    assert response.status_code == 400
    assert "session has expired" in response.get_json()["error"]


def test_post_with_wrong_csrf_token_is_rejected(api_client):
    response = api_client.client.post("/api/evaluations", json={}, headers={"X-CSRF-Token": "forged"})
    assert response.status_code == 400


def test_login_form_without_csrf_token_is_rejected(anon_client):
    anon_client.get("/login")
    response = anon_client.client.post("/login", data={"email": "a@b.co", "password": "password123"})
    assert response.status_code == 302                       # sent back to the form
    assert anon_client.get("/api/me").status_code == 401
