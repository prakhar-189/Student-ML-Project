# app.py
# =============================================
# Main Flask Application for ML Prediction.
# Serves an HTML form (/, /predictdata) and a JSON API (/predict, /health).
# =============================================

import logging
import os

from flask import Flask, jsonify, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

logger = logging.getLogger(__name__)

# Allowed categorical values (kept in sync with the training data / form).
CATEGORICAL_OPTIONS = {
    "gender": {"male", "female"},
    "race_ethnicity": {"group A", "group B", "group C", "group D", "group E"},
    "parental_level_of_education": {
        "associate's degree", "bachelor's degree", "high school",
        "master's degree", "some college", "some high school",
    },
    "lunch": {"free/reduced", "standard"},
    "test_preparation_course": {"none", "completed"},
}
NUMERIC_FIELDS = ("reading_score", "writing_score")


def _validate(payload):
    """Validates a raw input mapping. Returns (CustomData, None) or (None, error_message)."""
    cleaned = {}

    for field, allowed in CATEGORICAL_OPTIONS.items():
        value = (payload.get(field) or "").strip()
        if not value:
            return None, f"'{field}' is required."
        if value not in allowed:
            return None, f"'{field}' must be one of: {', '.join(sorted(allowed))}."
        cleaned[field] = value

    for field in NUMERIC_FIELDS:
        raw = payload.get(field)
        if raw is None or str(raw).strip() == "":
            return None, f"'{field}' is required."
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None, f"'{field}' must be a number between 0 and 100."
        if not 0 <= score <= 100:
            return None, f"'{field}' must be between 0 and 100."
        cleaned[field] = score

    data = CustomData(
        gender=cleaned["gender"],
        race_ethnicity=cleaned["race_ethnicity"],
        parental_level_of_education=cleaned["parental_level_of_education"],
        lunch=cleaned["lunch"],
        test_preparation_course=cleaned["test_preparation_course"],
        writing_score=cleaned["writing_score"],
        reading_score=cleaned["reading_score"],
    )
    return data, None


def _predict(data):
    """Runs the prediction pipeline and returns a rounded score."""
    prediction = PredictPipeline().predict(data.get_data_as_dataframe())
    return round(float(prediction[0]), 2)


@app.route("/")
def index():
    return render_template("home.html", results=None)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", results=None)

    data, error = _validate(request.form)
    if error:
        return render_template("home.html", results=None, error=error), 400
    try:
        return render_template("home.html", results=_predict(data))
    except Exception:
        # Log the real error server-side; never leak internals to the user.
        logger.exception("Prediction failed")
        return render_template(
            "home.html", results=None, error="Prediction failed. Please try again."
        ), 500


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON prediction endpoint: accepts the same fields and returns {"math_score": ...}."""
    payload = request.get_json(silent=True) or {}
    data, error = _validate(payload)
    if error:
        return jsonify({"error": error}), 400
    try:
        return jsonify({"math_score": _predict(data)})
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
