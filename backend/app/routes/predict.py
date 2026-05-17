# predict.py - runs (mock) inference. ML logic lives in src/aircraft_nlp.

from flask import Blueprint, jsonify, request

from aircraft_nlp.service import run_prediction

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Body must include a non-empty 'rows' list."}), 400

    return jsonify(run_prediction(rows))
