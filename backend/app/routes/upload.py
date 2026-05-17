# upload.py - accepts a CSV file, validates it, returns parsed rows.
# Thin route: parsing/validation only, no ML logic.

import csv
import io

from flask import Blueprint, jsonify, request

upload_bp = Blueprint("upload", __name__)

REQUIRED_COLUMNS = ["Discrepancy"]


@upload_bp.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"valid": False, "errors": ["No file part in request."]}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"valid": False, "errors": ["No file selected."]}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"valid": False, "errors": ["File must be a .csv"]}), 400

    try:
        text = file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        return jsonify({"valid": False, "errors": ["File is not valid UTF-8 text."]}), 400

    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    errors = []

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")

    rows = []
    if not missing:
        for i, row in enumerate(reader):
            row["id"] = i
            rows.append(row)
        if not rows:
            errors.append("File has headers but no data rows.")

    valid = not errors
    return jsonify(
        {
            "valid": valid,
            "errors": errors,
            "columns": columns,
            "rows": rows,
        }
    ), (200 if valid else 400)
