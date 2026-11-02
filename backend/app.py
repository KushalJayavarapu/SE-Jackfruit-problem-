"""
Web-Based ML Model Evaluator — Flask application.

Pages are rendered with Jinja; the page scripts call the JSON API under /api.
Every route except login, registration and static files needs a signed-in
user, and every model, dataset and evaluation is scoped to its owner (NFR-02).
"""
import datetime
import secrets
from functools import wraps
from urllib.parse import urlparse

import click
from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import config
from services import (admin_service, auth_service, dataset_service, evaluation_service,
                      metrics, model_service, report_service, visualization_service)
from services.errors import ForbiddenError, NotFoundError, ValidationError

app = Flask(__name__)
app.config.update(
    SECRET_KEY=config.secret_key(),
    MAX_CONTENT_LENGTH=110 * 1024 * 1024,  # model limit plus room for form data
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=config.SESSION_HOURS),
)

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def is_api_request():
    return request.path.startswith("/api/")


def json_error(message, status):
    return jsonify({"error": message}), status


# ---------------------------------------------------------------- sessions, CSRF

@app.before_request
def load_user_and_check_csrf():
    # Read the user from the database on every request, so a suspension or a
    # role change takes effect immediately (FR-08).
    g.user = auth_service.get_user(session.get("user_id"))
    if g.user and g.user["status"] != "active":
        session.clear()
        g.user = None

    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)

    if request.method in UNSAFE_METHODS:
        sent = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
        if not sent or not secrets.compare_digest(sent, session["csrf_token"]):
            if is_api_request():
                return json_error("Your session has expired. Refresh the page and try again.", 400)
            flash("Your session has expired. Please try again.", "error")
            return redirect(url_for("index") if request.path == "/logout" else request.path)


@app.context_processor
def template_globals():
    return {"current_user": g.get("user"), "csrf_token": session.get("csrf_token", "")}


def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if g.user is None:
            if is_api_request():
                return json_error("Please log in to continue.", 401)
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)
    return wrapper


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if g.user["role"] != "admin":
            if is_api_request():
                return json_error("This area is only for Admins.", 403)
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def home_for(user):
    return url_for("admin_users_page") if user["role"] == "admin" else url_for("dashboard")


def safe_next(target):
    """Only follow `next` links that stay on this site."""
    if target and urlparse(target).netloc == "" and target.startswith("/") and not target.startswith("//"):
        return target
    return None


# ---------------------------------------------------------------- error handling (BL-07)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return json_error("The uploaded file is too large. Please choose a smaller file.", 413)


@app.errorhandler(HTTPException)
def handle_http_error(error):
    if is_api_request() or request.method != "GET":
        return json_error(error.description, error.code)
    return render_template("error.html", code=error.code, message=error.description), error.code


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.exception("Unexpected application error")
    message = "Something went wrong. Please check your inputs and try again."
    if is_api_request():
        return json_error(message, 500)
    return render_template("error.html", code=500, message=message), 500


def service_error(exc):
    if isinstance(exc, NotFoundError):
        return json_error(str(exc), 404)
    if isinstance(exc, ForbiddenError):
        return json_error(str(exc), 403)
    return json_error(str(exc), 400)


# ---------------------------------------------------------------- login and registration (FR-01)

@app.route("/")
def index():
    if g.user is None:
        return redirect(url_for("login"))
    return redirect(home_for(g.user))


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(home_for(g.user))
    email = ""
    if request.method == "POST":
        email = request.form.get("email", "")
        try:
            user = auth_service.authenticate(email, request.form.get("password", ""))
        except ValidationError as exc:
            flash(str(exc), "error")
            return render_template("login.html", email=email), 400
        session.clear()
        session.permanent = True
        session["user_id"] = user["id"]
        return redirect(safe_next(request.args.get("next")) or home_for(user))
    return render_template("login.html", email=email)


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(home_for(g.user))
    form = {"name": "", "email": ""}
    if request.method == "POST":
        form = {"name": request.form.get("name", ""), "email": request.form.get("email", "")}
        try:
            user = auth_service.register(form["name"], form["email"], request.form.get("password", ""),
                                         request.form.get("confirm_password", ""))
        except ValidationError as exc:
            flash(str(exc), "error")
            return render_template("register.html", form=form), 400
        session.clear()
        session.permanent = True
        session["user_id"] = user["id"]
        flash(f"Welcome, {user['name']}. Your Analyst account is ready.", "success")
        return redirect(url_for("dashboard"))
    return render_template("register.html", form=form)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You have logged out.", "success")
    return redirect(url_for("login"))


@app.route("/api/me")
@login_required
def api_me():
    return jsonify(g.user)


# ---------------------------------------------------------------- workspace pages

@app.route("/dashboard")
@login_required
def dashboard():
    uid = g.user["id"]
    models = model_service.list_models(uid)
    datasets = dataset_service.list_datasets(uid)
    evaluations = evaluation_service.list_evaluations(uid)
    return render_template(
        "dashboard.html", active="dashboard",
        models=models, datasets=datasets, evaluations=evaluations[:5],
        evaluation_count=len(evaluations),
        has_valid_model=any(m["validation_status"] == "valid" for m in models),
        has_valid_dataset=any(d["validation_status"] == "valid" for d in datasets),
    )


@app.route("/models")
@login_required
def models_page():
    return render_template("models.html", active="models", max_mb=config.MAX_MODEL_SIZE_BYTES // (1024 * 1024))


@app.route("/datasets")
@login_required
def datasets_page():
    return render_template("datasets.html", active="datasets", max_rows=config.MAX_DATASET_ROWS)


@app.route("/evaluate")
@login_required
def evaluate_page():
    return render_template("evaluate.html", active="evaluate")


@app.route("/evaluations")
@login_required
def history_page():
    return render_template("history.html", active="history")


@app.route("/evaluations/<evaluation_id>")
@login_required
def evaluation_page(evaluation_id):
    try:
        evaluation = evaluation_service.get_evaluation(evaluation_id, g.user["id"])
    except NotFoundError as exc:
        return render_template("error.html", code=404, message=str(exc), active="history"), 404
    return render_template("evaluation.html", active="history", evaluation=evaluation,
                           metric_info=metrics.METRIC_INFO,
                           metric_keys=metrics.metric_keys(evaluation["task_type"]))


@app.route("/compare")
@login_required
def compare_page():
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    try:
        comparison = evaluation_service.compare(ids, g.user["id"])
        error = None
    except ValidationError as exc:
        comparison, error = None, str(exc)
    return render_template("compare.html", active="history", comparison=comparison, error=error,
                           ids=",".join(ids), metric_info=metrics.METRIC_INFO)


# ---------------------------------------------------------------- models API (BL-01, BL-02)

@app.route("/api/models", methods=["GET"])
@login_required
def api_list_models():
    return jsonify(model_service.list_models(g.user["id"])), 200


@app.route("/api/models/upload", methods=["POST"])
@login_required
def api_upload_model():
    try:
        record = model_service.upload_model(request.files.get("model_file"), g.user["id"])
    except ValidationError as exc:
        return json_error(str(exc), 400)
    return jsonify(record), 201


@app.route("/api/models/<model_id>/validate", methods=["POST"])
@login_required
def api_validate_model(model_id):
    try:
        return jsonify(model_service.validate_model(model_id, g.user["id"])), 200
    except ValidationError as exc:
        return json_error(str(exc), 404)


@app.route("/api/models/<model_id>", methods=["DELETE"])
@login_required
def api_delete_model(model_id):
    try:
        model_service.delete_model(model_id, g.user["id"])
    except ValidationError as exc:
        return service_error(exc)
    return jsonify({"deleted": model_id}), 200


# ---------------------------------------------------------------- datasets API (BL-03, BL-04)

@app.route("/api/datasets", methods=["GET"])
@login_required
def api_list_datasets():
    return jsonify(dataset_service.list_datasets(g.user["id"])), 200


@app.route("/api/datasets/upload", methods=["POST"])
@login_required
def api_upload_dataset():
    try:
        record = dataset_service.upload_dataset(
            request.files.get("dataset_file"), request.form.get("target_variable", ""), g.user["id"])
    except ValidationError as exc:
        return json_error(str(exc), 400)
    return jsonify(record), 201


@app.route("/api/datasets/<dataset_id>/validate", methods=["POST"])
@login_required
def api_validate_dataset(dataset_id):
    try:
        return jsonify(dataset_service.validate_dataset(dataset_id, g.user["id"])), 200
    except ValidationError as exc:
        return json_error(str(exc), 404)


@app.route("/api/datasets/<dataset_id>", methods=["DELETE"])
@login_required
def api_delete_dataset(dataset_id):
    try:
        dataset_service.delete_dataset(dataset_id, g.user["id"])
    except ValidationError as exc:
        return service_error(exc)
    return jsonify({"deleted": dataset_id}), 200


# ---------------------------------------------------------------- evaluations API (BL-05, FR-05..07)

@app.route("/api/evaluations", methods=["GET"])
@login_required
def api_list_evaluations():
    return jsonify(evaluation_service.list_evaluations(g.user["id"])), 200


@app.route("/api/evaluations", methods=["POST"])
@login_required
def api_run_evaluation():
    if not request.is_json:
        return json_error("Send evaluation details as JSON with model_id, dataset_id, and task_type.", 400)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return json_error("The request body must be a valid JSON object.", 400)

    required_fields = ("model_id", "dataset_id", "task_type")
    missing = [f for f in required_fields if not isinstance(body.get(f), str) or not body[f].strip()]
    if missing:
        return json_error(f"Missing or invalid required field(s): {', '.join(missing)}.", 400)

    try:
        record = evaluation_service.evaluate_model(
            body["model_id"], body["dataset_id"], body["task_type"], g.user["id"])
    except ValidationError as exc:
        return json_error(str(exc), 400)
    return jsonify(record), 201


@app.route("/api/evaluations/<evaluation_id>", methods=["GET"])
@login_required
def api_get_evaluation(evaluation_id):
    try:
        return jsonify(evaluation_service.get_evaluation(evaluation_id, g.user["id"])), 200
    except ValidationError as exc:
        return service_error(exc)


@app.route("/api/evaluations/<evaluation_id>/charts/<kind>.png", methods=["GET"])
@login_required
def api_evaluation_chart(evaluation_id, kind):
    record = evaluation_service.get_evaluation_record(evaluation_id, g.user["id"])
    if record is None:
        return json_error("This evaluation is not available. It may have been removed.", 404)
    png = visualization_service.render(kind, record["charts"])
    if png is None:
        return json_error("This chart is not available for this evaluation.", 404)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "private, max-age=3600"})


def _ids_arg():
    return [i for i in request.args.get("ids", "").split(",") if i.strip()]


@app.route("/api/compare", methods=["GET"])
@login_required
def api_compare():
    try:
        return jsonify(evaluation_service.compare(_ids_arg(), g.user["id"])), 200
    except ValidationError as exc:
        return service_error(exc)


@app.route("/api/compare/chart.png", methods=["GET"])
@login_required
def api_compare_chart():
    try:
        comparison = evaluation_service.compare(_ids_arg(), g.user["id"])
    except ValidationError as exc:
        return service_error(exc)
    png = visualization_service.render_comparison(comparison["evaluations"], comparison["metric_keys"])
    return Response(png, mimetype="image/png", headers={"Cache-Control": "private, max-age=600"})


@app.route("/api/reports", methods=["GET"])
@login_required
def api_report():
    try:
        records = evaluation_service.get_many(_ids_arg(), g.user["id"])
    except ValidationError as exc:
        return service_error(exc)
    pdf = report_service.build_pdf(records, g.user)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
    return Response(pdf, mimetype="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="evaluation-report-{stamp}.pdf"'})


# ---------------------------------------------------------------- admin (FR-08)

@app.route("/admin/users")
@admin_required
def admin_users_page():
    return render_template("admin_users.html", active="admin_users", min_password=config.MIN_PASSWORD_LENGTH)


@app.route("/admin/audit")
@admin_required
def admin_audit_page():
    return render_template("admin_audit.html", active="admin_audit", entries=admin_service.list_audit_log())


@app.route("/api/admin/users", methods=["GET"])
@admin_required
def api_admin_users():
    return jsonify(admin_service.list_users()), 200


@app.route("/api/admin/users", methods=["POST"])
@admin_required
def api_admin_create_user():
    body = request.get_json(silent=True) or {}
    try:
        user = admin_service.create_account(g.user, body.get("name"), body.get("email"),
                                            body.get("password"), body.get("role", "analyst"))
    except ValidationError as exc:
        return service_error(exc)
    return jsonify(user), 201


@app.route("/api/admin/users/<user_id>/status", methods=["POST"])
@admin_required
def api_admin_set_status(user_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(admin_service.set_status(g.user, user_id, body.get("status"))), 200
    except ValidationError as exc:
        return service_error(exc)


@app.route("/api/admin/users/<user_id>/role", methods=["POST"])
@admin_required
def api_admin_set_role(user_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(admin_service.set_role(g.user, user_id, body.get("role"))), 200
    except ValidationError as exc:
        return service_error(exc)


@app.route("/api/admin/audit", methods=["GET"])
@admin_required
def api_admin_audit():
    return jsonify(admin_service.list_audit_log()), 200


# ---------------------------------------------------------------- command line

@app.cli.command("create-admin")
@click.option("--name", prompt="Name")
@click.option("--email", prompt="Email")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
def create_admin_command(name, email, password):
    """Create an Admin account. Use this once to make the first Admin."""
    try:
        user = admin_service.create_account(None, name, email, password, "admin")
    except ValidationError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Admin account created for {user['email']}. Log in at http://127.0.0.1:5000/login")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
