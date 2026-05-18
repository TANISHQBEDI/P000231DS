# history.py - lists / fetches archived prediction records.

from flask import Blueprint, jsonify

from app.services.umec_storage import get_history_item, list_history

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history", methods=["GET"])
def history():
    return jsonify({"items": list_history()})


@history_bp.route("/api/history/<record_id>", methods=["GET"])
def history_item(record_id):
    item = get_history_item(record_id)
    if item is None:
        return jsonify({"error": "Record not found."}), 404
    return jsonify(item)
