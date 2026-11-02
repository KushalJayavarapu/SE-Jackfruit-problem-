"""
FR-01 / UC-01 / UC-02: registration and login.

Registration only ever creates Analyst accounts. Admin accounts are created by
an existing Admin (admin_service.create_account) or, for the very first one,
with the `flask create-admin` command.
"""
import re
import uuid

from werkzeug.security import check_password_hash, generate_password_hash

import config
from services import db
from services.errors import ValidationError
from services.timeutil import now_iso

ROLES = ("analyst", "admin")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValidationError):
    """Login refused; the message is safe to show."""


def public_user(user):
    """A user record without the password hash."""
    if user is None:
        return None
    return {k: v for k, v in user.items() if k != "password_hash"}


def _clean_registration(name, email, password, confirm_password):
    name = (name or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    errors = []
    if not name:
        errors.append("Enter your name.")
    elif len(name) > 80:
        errors.append("Name must be 80 characters or fewer.")
    if not email:
        errors.append("Enter your email address.")
    elif not EMAIL_PATTERN.match(email) or len(email) > 254:
        errors.append("Enter a valid email address, like name@example.com.")
    if len(password) < config.MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters.")
    if confirm_password is not None and password != confirm_password:
        errors.append("The two passwords do not match.")
    if errors:
        raise ValidationError(" ".join(errors))
    return name, email, password


def email_taken(email):
    return db.fetch_one("users", "email = ?", ((email or "").strip().lower(),)) is not None


def _create(name, email, password, role, confirm_password=None):
    name, email, password = _clean_registration(name, email, password, confirm_password)
    if role not in ROLES:
        raise ValidationError("Role must be 'analyst' or 'admin'.")
    if email_taken(email):
        raise ValidationError("An account with this email already exists. Log in instead.")
    user = {
        "id": str(uuid.uuid4()),
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "role": role,
        "status": "active",
        "created_at": now_iso(),
        "last_login_at": None,
    }
    db.insert("users", user)
    return user


def register(name, email, password, confirm_password):
    """Open self-registration. Always creates an Analyst (FR-01)."""
    return public_user(_create(name, email, password, "analyst", confirm_password))


def authenticate(email, password):
    email = (email or "").strip().lower()
    if not email or not password:
        raise AuthError("Enter your email and password.")
    user = db.fetch_one("users", "email = ?", (email,))
    if user is None or not check_password_hash(user["password_hash"], password):
        raise AuthError("Incorrect email or password.")
    if user["status"] != "active":
        raise AuthError("This account is suspended. Contact an administrator to reactivate it.")
    db.update("users", user["id"], {"last_login_at": now_iso()})
    return public_user(user)


def get_user(user_id):
    if not user_id:
        return None
    return public_user(db.fetch_one("users", "id = ?", (user_id,)))
