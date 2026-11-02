"""
FR-08 / UC-12 / UC-13: account and role management, with an audit log.

Rules:
- an Admin cannot suspend, reactivate or change the role of their own account;
- the last active Admin can never be suspended or demoted;
- asking for a state the account is already in changes nothing and is refused;
- every change is written to the audit log with who, what and when.
"""
from services import auth_service, db
from services.errors import ForbiddenError, NotFoundError, ValidationError
from services.timeutil import now_iso

STATUSES = ("active", "suspended")


def _log(actor, target, action, old_value, new_value):
    db.insert("audit_log", {
        "actor_id": actor["id"] if actor else None,
        "actor_email": actor["email"] if actor else "command line",
        "target_id": target["id"],
        "target_email": target["email"],
        "action": action,
        "old_value": old_value,
        "new_value": new_value,
        "created_at": now_iso(),
    })


def _require_admin(actor):
    if not actor or actor.get("role") != "admin" or actor.get("status") != "active":
        raise ForbiddenError("Only an active Admin can manage accounts.")


def _target(user_id):
    user = db.fetch_one("users", "id = ?", (user_id,))
    if user is None:
        raise NotFoundError("No such account.")
    return user


def _active_admin_count():
    return db.count("users", "role = 'admin' AND status = 'active'")


def list_users():
    return [auth_service.public_user(u) for u in db.fetch_all("users", order="created_at")]


def create_account(actor, name, email, password, role):
    """An Admin creates an account. This is the only way to create a new Admin (FR-01)."""
    if actor is not None:
        _require_admin(actor)
    user = auth_service._create(name, email, password, role)
    _log(actor, user, "create_account", None, role)
    return auth_service.public_user(user)


def set_status(actor, user_id, status):
    _require_admin(actor)
    if status not in STATUSES:
        raise ValidationError("Status must be 'active' or 'suspended'.")
    target = _target(user_id)
    if target["id"] == actor["id"]:
        raise ForbiddenError("You cannot change the status of your own account.")
    if target["status"] == status:
        raise ValidationError(f"This account is already {status}.")
    if status == "suspended" and target["role"] == "admin" and _active_admin_count() <= 1:
        raise ForbiddenError("This is the last active Admin, so it cannot be suspended.")

    db.update("users", target["id"], {"status": status})
    _log(actor, target, "suspend" if status == "suspended" else "reactivate", target["status"], status)
    return auth_service.get_user(target["id"])


def set_role(actor, user_id, role):
    _require_admin(actor)
    if role not in auth_service.ROLES:
        raise ValidationError("Role must be 'analyst' or 'admin'.")
    target = _target(user_id)
    if target["id"] == actor["id"]:
        raise ForbiddenError("You cannot change your own role.")
    if target["role"] == role:
        raise ValidationError(f"This account is already an {role.capitalize()}.")
    if (target["role"] == "admin" and target["status"] == "active"
            and role == "analyst" and _active_admin_count() <= 1):
        raise ForbiddenError("This is the last active Admin, so it cannot be changed to Analyst.")

    db.update("users", target["id"], {"role": role})
    _log(actor, target, "role_change", target["role"], role)
    return auth_service.get_user(target["id"])


def list_audit_log(limit=500):
    return db.fetch_all("audit_log", order="id DESC LIMIT %d" % int(limit))
