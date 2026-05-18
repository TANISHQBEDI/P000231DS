# train.py - triggers UMEC training using configs/core.

from flask import Blueprint, jsonify, request

from app.services.umec_training import run_training

train_bp = Blueprint("train", __name__)


@train_bp.route("/api/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    result = run_training(
        dataset_meta=payload.get("dataset_meta"),
        feedback=payload.get("feedback"),
    )
    return jsonify(result)
