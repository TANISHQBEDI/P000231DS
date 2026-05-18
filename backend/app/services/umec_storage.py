"""Storage + audit trail for UMEC predictions."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from umec.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = REPO_ROOT / "history"
AUDIT_LOG = HISTORY_DIR / "maintenance_log.jsonl"

_HISTORY: list[dict] = []


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _get_config_dir() -> Path:
    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _model_version() -> str:
    try:
        cfg = load_config(_get_config_dir())
        return f"umec-{Path(cfg.project.model_dir).name}"
    except Exception:
        return "umec-unknown"


def _audit(entry: dict) -> None:
    _ensure_dir()
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def save_predictions(
    records: list[dict],
    user: str = "anonymous",
    before: list[dict] | None = None,
) -> dict:
    _ensure_dir()
    rec_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    model_version = _model_version()

    record = {
        "id": rec_id,
        "timestamp": ts,
        "model_version": model_version,
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
            "model_version": model_version,
            "user": user,
            "action": "save_predictions",
            "num_records": record["num_records"],
            "file": out_file.name,
        }
    )

    return {
        "id": rec_id,
        "timestamp": ts,
        "model_version": model_version,
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
