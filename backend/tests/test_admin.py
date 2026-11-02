"""Admin account and role management (FR-08, UC-12, UC-13) and the audit log."""
import pytest

from services import admin_service, db


def status(client, user_id, value):
    return client.post(f"/api/admin/users/{user_id}/status", json={"status": value})


def role(client, user_id, value):
    return client.post(f"/api/admin/users/{user_id}/role", json={"role": value})


# ---------------- who may use the admin area ----------------

@pytest.mark.parametrize("path", ["/api/admin/users", "/api/admin/audit"])
def test_analyst_cannot_use_admin_api(api_client, path):
    response = api_client.get(path)
    assert response.status_code == 403
    assert response.get_json() == {"error": "This area is only for Admins."}


@pytest.mark.parametrize("path", ["/admin/users", "/admin/audit"])
def test_analyst_cannot_open_admin_pages(api_client, path):
    assert api_client.get(path).status_code == 403


def test_analyst_cannot_change_roles_or_statuses_by_calling_the_api(api_client, make_user):
    _, other = make_user("other@example.com")
    assert role(api_client, other["id"], "admin").status_code == 403
    assert status(api_client, other["id"], "suspended").status_code == 403
    assert db.fetch_one("users", "id = ?", (other["id"],))["role"] == "analyst"


def test_analyst_cannot_make_themselves_admin(api_client):
    me = api_client.get("/api/me").get_json()
    assert role(api_client, me["id"], "admin").status_code == 403
    assert api_client.get("/api/me").get_json()["role"] == "analyst"


def test_analyst_cannot_create_accounts(api_client):
    response = api_client.post("/api/admin/users", json={"name": "X", "email": "x@example.com", "password": "password123", "role": "admin"})
    assert response.status_code == 403


# ---------------- 12_UT_01: list accounts ----------------

def test_12_UT_01_admin_sees_all_accounts_with_status_and_role(admin_client, make_user):
    make_user("one@example.com")
    users = admin_client.get("/api/admin/users").get_json()
    emails = {u["email"]: u for u in users}
    assert {"boss@example.com", "one@example.com"} <= set(emails)
    assert emails["boss@example.com"]["role"] == "admin"
    assert all("password_hash" not in u for u in users)
    assert admin_client.get("/admin/users").status_code == 200


# ---------------- 12_UT_02 / 12_UT_03: suspend and reactivate ----------------

def test_12_UT_02_admin_can_suspend_and_12_UT_03_reactivate(admin_client, make_user):
    _, user = make_user("target@example.com")

    suspended = status(admin_client, user["id"], "suspended")
    assert suspended.status_code == 200 and suspended.get_json()["status"] == "suspended"

    reactivated = status(admin_client, user["id"], "active")
    assert reactivated.status_code == 200 and reactivated.get_json()["status"] == "active"


def test_12_ST_02_setting_the_state_an_account_already_has_is_refused(admin_client, make_user):
    _, user = make_user("same@example.com")
    response = status(admin_client, user["id"], "active")
    assert response.status_code == 400 and "already active" in response.get_json()["error"]


def test_12_IT_01_admin_cannot_suspend_their_own_account(admin_client, make_user):
    make_user("backup-admin@example.com", role="admin")      # not the last admin: the rule is about "own account"
    response = status(admin_client, admin_client.user["id"], "suspended")
    assert response.status_code == 403
    assert "your own account" in response.get_json()["error"]


def test_12_IT_02_last_active_admin_cannot_be_suspended(admin_client, make_user):
    other_client, other = make_user("second-admin@example.com", role="admin")
    assert status(admin_client, other["id"], "suspended").status_code == 200       # two admins: fine
    # Now `boss` is the only active admin. A suspended admin cannot act, so use the
    # service directly to try to suspend the last one from a (hypothetical) second actor.
    actor = {"id": other["id"], "email": other["email"], "role": "admin", "status": "active"}
    with pytest.raises(admin_service.ForbiddenError, match="last active Admin"):
        admin_service.set_status(actor, admin_client.user["id"], "suspended")


def test_suspended_admin_cannot_use_admin_functions(admin_client, make_user):
    other_client, other = make_user("temp-admin@example.com", role="admin")
    status(admin_client, other["id"], "suspended")
    assert other_client.get("/api/admin/users").status_code == 401


def test_status_must_be_active_or_suspended(admin_client, make_user):
    _, user = make_user("bad@example.com")
    assert status(admin_client, user["id"], "deleted").status_code == 400


def test_unknown_account_returns_404(admin_client):
    assert status(admin_client, "no-such-id", "suspended").status_code == 404
    assert role(admin_client, "no-such-id", "admin").status_code == 404


# ---------------- 13_*: roles ----------------

def test_13_UT_03_admin_can_make_an_analyst_an_admin_and_back(admin_client, make_user):
    user_client, user = make_user("promote@example.com")
    assert user_client.get("/api/admin/users").status_code == 403

    assert role(admin_client, user["id"], "admin").get_json()["role"] == "admin"
    assert user_client.get("/api/admin/users").status_code == 200             # takes effect at once

    assert role(admin_client, user["id"], "analyst").get_json()["role"] == "analyst"
    assert user_client.get("/api/admin/users").status_code == 403


def test_13_IT_03_admin_cannot_change_their_own_role(admin_client, make_user):
    make_user("another-admin@example.com", role="admin")
    response = role(admin_client, admin_client.user["id"], "analyst")
    assert response.status_code == 403 and "your own role" in response.get_json()["error"]


def test_13_ST_02_last_active_admin_cannot_be_demoted(admin_client, make_user):
    other_client, other = make_user("deputy@example.com", role="admin")
    assert role(admin_client, other["id"], "analyst").status_code == 200
    actor = {"id": other["id"], "email": other["email"], "role": "admin", "status": "active"}
    with pytest.raises(admin_service.ForbiddenError, match="last active Admin"):
        admin_service.set_role(actor, admin_client.user["id"], "analyst")


def test_role_must_be_analyst_or_admin(admin_client, make_user):
    _, user = make_user("odd@example.com")
    assert role(admin_client, user["id"], "owner").status_code == 400


def test_setting_the_role_an_account_already_has_is_refused(admin_client, make_user):
    _, user = make_user("noop@example.com")
    response = role(admin_client, user["id"], "analyst")
    assert response.status_code == 400 and "already an Analyst" in response.get_json()["error"]


def test_only_an_admin_can_create_an_admin_account(admin_client, new_client):
    created = admin_client.post("/api/admin/users", json={"name": "New Admin", "email": "newadmin@example.com", "password": "password123", "role": "admin"})
    assert created.status_code == 201 and created.get_json()["role"] == "admin"
    assert new_client().login("newadmin@example.com", "password123").status_code == 302


def test_admin_account_creation_validates_input(admin_client):
    response = admin_client.post("/api/admin/users", json={"name": "", "email": "nope", "password": "x", "role": "analyst"})
    assert response.status_code == 400
    duplicate = admin_client.post("/api/admin/users", json={"name": "Dup", "email": "boss@example.com", "password": "password123", "role": "analyst"})
    assert duplicate.status_code == 400 and "already exists" in duplicate.get_json()["error"]


# ---------------- audit log ----------------

def test_13_IT_02_every_change_is_logged_with_who_what_and_when(admin_client, make_user):
    _, user = make_user("logged@example.com")
    status(admin_client, user["id"], "suspended")
    status(admin_client, user["id"], "active")
    role(admin_client, user["id"], "admin")

    log = admin_client.get("/api/admin/audit").get_json()
    actions = [(e["action"], e["old_value"], e["new_value"]) for e in log]

    assert actions[0] == ("role_change", "analyst", "admin")           # newest first
    assert ("reactivate", "suspended", "active") in actions
    assert ("suspend", "active", "suspended") in actions
    for entry in log[:3]:
        assert entry["actor_email"] == "boss@example.com"
        assert entry["actor_id"] == admin_client.user["id"]
        assert entry["target_email"] == "logged@example.com"
        assert entry["created_at"].endswith("Z")


def test_refused_changes_are_not_logged(admin_client, make_user):
    make_user("spare-admin@example.com", role="admin")
    before = db.count("audit_log")
    status(admin_client, admin_client.user["id"], "suspended")          # refused: own account
    status(admin_client, "no-such-id", "suspended")                     # refused: unknown
    assert db.count("audit_log") == before


def test_account_creation_is_logged(admin_client):
    admin_client.post("/api/admin/users", json={"name": "Nu", "email": "nu@example.com", "password": "password123", "role": "analyst"})
    entry = admin_client.get("/api/admin/audit").get_json()[0]
    assert entry["action"] == "create_account" and entry["new_value"] == "analyst"


def test_audit_log_page_lists_entries(admin_client, make_user):
    _, user = make_user("shown@example.com")
    status(admin_client, user["id"], "suspended")
    page = admin_client.get("/admin/audit").get_data(as_text=True)
    assert "shown@example.com" in page and "Suspended" in page


def test_first_admin_can_be_created_from_the_command_line(isolated_storage):
    import app as app_module
    runner = app_module.app.test_cli_runner()
    result = runner.invoke(args=["create-admin", "--name", "Root", "--email", "root@example.com",
                                 "--password", "password123"])
    assert result.exit_code == 0, result.output
    user = db.fetch_one("users", "email = ?", ("root@example.com",))
    assert user["role"] == "admin"
    assert db.fetch_all("audit_log")[-1]["actor_email"] == "command line"
