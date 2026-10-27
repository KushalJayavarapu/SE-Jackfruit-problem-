
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from services import model_service, dataset_service, evaluation_service
from services.model_service import ValidationError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 110 * 1024 * 1024  # 110 MB upload limit


# ---------- BL-07: Global error handling ----------

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    return jsonify({
        "error": "The uploaded file is too large. Please choose a smaller file."
    }), 413


@app.errorhandler(HTTPException)
def handle_http_error(error):
    return jsonify({
        "error": error.description
    }), error.code


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    app.logger.exception("Unexpected application error")
    return jsonify({
        "error": "Something went wrong. Please check your inputs and try again."
    }), 500


# ---------- Home page ----------

@app.route("/")
def index():
    return render_template(
        "index.html",
        models=model_service.list_models(),
        datasets=dataset_service.list_datasets(),
    )


# ---------- BL-01 / BL-02: Models ----------

@app.route("/api/models", methods=["GET"])
def api_list_models():
    return jsonify(model_service.list_models()), 200


@app.route("/api/models/upload", methods=["POST"])
def api_upload_model():
    file_storage = request.files.get("model_file")

    try:
        record = model_service.upload_model(file_storage)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(record), 201


@app.route("/api/models/<model_id>/validate", methods=["POST"])
def api_validate_model(model_id):
    try:
        record = model_service.validate_model(model_id)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(record), 200


# ---------- BL-03 / BL-04: Datasets ----------

@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    return jsonify(dataset_service.list_datasets()), 200


@app.route("/api/datasets/upload", methods=["POST"])
def api_upload_dataset():
    file_storage = request.files.get("dataset_file")
    target_variable = request.form.get("target_variable", "")

    try:
        record = dataset_service.upload_dataset(
            file_storage,
            target_variable,
        )
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(record), 201


@app.route("/api/datasets/<dataset_id>/validate", methods=["POST"])
def api_validate_dataset(dataset_id):
    try:
        record = dataset_service.validate_dataset(dataset_id)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 404

    return jsonify(record), 200


# ---------- BL-05: Evaluation ----------

@app.route("/api/evaluations", methods=["GET"])
def api_list_evaluations():
    return jsonify(evaluation_service.list_evaluations()), 200


@app.route("/api/evaluations", methods=["POST"])
def api_run_evaluation():
    # Reject non-JSON requests.
    if not request.is_json:
        return jsonify({
            "error": (
                "Send evaluation details as JSON with "
                "model_id, dataset_id, and task_type."
            )
        }), 400

    body = request.get_json(silent=True)

    # Reject malformed JSON or a JSON body that isn't an object.
    if not isinstance(body, dict):
        return jsonify({
            "error": "The request body must be a valid JSON object."
        }), 400

    # Check that all required fields are non-empty strings.
    required_fields = ("model_id", "dataset_id", "task_type")
    missing_fields = [
        field
        for field in required_fields
        if not isinstance(body.get(field), str)
        or not body[field].strip()
    ]

    if missing_fields:
        return jsonify({
            "error": (
                "Missing or invalid required field(s): "
                + ", ".join(missing_fields)
                + "."
            )
        }), 400

    try:
        record = evaluation_service.evaluate_model(
            body["model_id"],
            body["dataset_id"],
            body["task_type"],
        )
    except ValidationError as exc:
        # Invalid IDs, incompatible inputs, or failed validation.
        return jsonify({"error": str(exc)}), 400

    return jsonify(record), 201


if __name__ == "__main__":
    app.run(debug=True, port=5000)
