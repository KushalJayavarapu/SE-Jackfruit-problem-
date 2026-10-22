from flask import Flask, request, jsonify, render_template

from services import model_service, dataset_service
from services.model_service import ValidationError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 110 * 1024 * 1024  # a little headroom over the 100 MB model cap


@app.route("/")
def index():
    return render_template(
        "index.html",
        models=model_service.list_models(),
        datasets=dataset_service.list_datasets(),
    )


# ---------- BL-01 / BL-02: models ----------

@app.route("/api/models", methods=["GET"])
def api_list_models():
    return jsonify(model_service.list_models())


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


# ---------- BL-03: datasets ----------

@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    return jsonify(dataset_service.list_datasets())


@app.route("/api/datasets/upload", methods=["POST"])
def api_upload_dataset():
    file_storage = request.files.get("dataset_file")
    target_variable = request.form.get("target_variable", "")
    try:
        record = dataset_service.upload_dataset(file_storage, target_variable)
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(record), 201


if __name__ == "__main__":
    app.run(debug=True, port=5000)
