# feedback.py - persists user-edited predictions + writes the audit trail.

from flask import Blueprint, jsonify, request

from app.services.umec_storage import save_predictions

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/api/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return jsonify({"error": "Body must include a non-empty 'records' list."}), 400

    saved = save_predictions(
        records=records,
        user=payload.get("user", "anonymous"),
        before=payload.get("before"),
    )
    return jsonify(saved), 201
