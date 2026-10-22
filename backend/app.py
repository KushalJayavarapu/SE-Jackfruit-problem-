from flask import Flask, request, jsonify, render_template

from services import model_service
from services.model_service import ValidationError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 110 * 1024 * 1024  # a little headroom over the 100 MB model cap


@app.route("/")
def index():
    return render_template("index.html", models=model_service.list_models())


# ---------- BL-01: models ----------

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
