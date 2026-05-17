"""Mock storage + audit wrapper.

Persists user-edited predictions with automated versioning (timestamp +
uuid), keeps an in-memory list (the real seam), writes one JSON file per
save into ``<repo>/history/``, and appends a Maintenance-Log line to
``<repo>/history/maintenance_log.jsonl``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .prediction import MODEL_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = REPO_ROOT / "history"
AUDIT_LOG = HISTORY_DIR / "maintenance_log.jsonl"

# Real seam: in-memory record list (survives within a running server).
_HISTORY: list[dict] = []


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _audit(entry: dict) -> None:
    _ensure_dir()
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def save_predictions(records: list[dict],
                     user: str = "anonymous",
                     before: list[dict] | None = None) -> dict:
    """Save edited predictions. ``before`` (optional) is the pre-edit state
    so the record stores a before/after diff for the audit trail."""
    _ensure_dir()
    rec_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    record = {
        "id": rec_id,
        "timestamp": ts,
        "model_version": MODEL_VERSION,
        "user": user,
        "num_records": len(records or []),
        "before": before or [],
        "after": records or [],
    }

    safe_ts = ts.replace(":", "-")
    out_file = HISTORY_DIR / f"{safe_ts}_{rec_id}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    _HISTORY.append(record)
    _audit(
        {
            "timestamp": ts,
            "id": rec_id,
            "model_version": MODEL_VERSION,
            "user": user,
            "action": "save_predictions",
            "num_records": record["num_records"],
            "file": out_file.name,
        }
    )

    return {
        "id": rec_id,
        "timestamp": ts,
        "model_version": MODEL_VERSION,
        "num_records": record["num_records"],
        "file": out_file.name,
    }


def _summary(record: dict) -> dict:
    return {
        "id": record["id"],
        "timestamp": record["timestamp"],
        "model_version": record["model_version"],
        "user": record["user"],
        "num_records": record["num_records"],
    }


def list_history() -> list[dict]:
    """Newest first. Falls back to disk if the in-memory list is empty
    (e.g. after a server restart)."""
    if _HISTORY:
        return [_summary(r) for r in reversed(_HISTORY)]

    if not HISTORY_DIR.exists():
        return []
    summaries = []
    for f in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            summaries.append(_summary(json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return summaries


def get_history_item(record_id: str) -> dict | None:
    for r in _HISTORY:
        if r["id"] == record_id:
            return r
    if not HISTORY_DIR.exists():
        return None
    for f in HISTORY_DIR.glob(f"*_{record_id}.json"):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None
