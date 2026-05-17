"""Mock prediction wrapper.

Contract (stable — real model must return this shape):
    run_prediction(rows) -> {
        "model_version": str,
        "generated_at": iso8601,
        "predictions": [
            {
                "row_id": int,
                "discrepancy": str,
                "predicted_condition": str,
                "confidence": float,        # 0..1
                "xai": {"keyword": str, "explanation": str},
                "low_confidence": bool,
            },
            ...
        ],
    }

Swap real inference into ``_mock_infer`` only; everything else stays.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MODEL_VERSION = "mock-0.1.0"
LOW_CONFIDENCE_THRESHOLD = 0.60

_LABELS_CACHE: list[str] | None = None


def _load_labels() -> list[str]:
    """Reuse the project's label mapping so mock output looks realistic."""
    global _LABELS_CACHE
    if _LABELS_CACHE is not None:
        return _LABELS_CACHE
    cfg = Path(__file__).resolve().parents[1] / "config" / "label_mappings.json"
    try:
        mapping = json.loads(cfg.read_text(encoding="utf-8"))
        labels = sorted(set(mapping.values()))
    except (OSError, ValueError):
        labels = ["faulty", "damaged", "corroded", "worn", "none"]
    _LABELS_CACHE = labels
    return labels


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _pick_keyword(text: str) -> str:
    words = re.findall(r"[A-Za-z]{4,}", text)
    if not words:
        return text.strip()[:20] or "n/a"
    # deterministic pick
    return words[_stable_hash(text) % len(words)].lower()


def _mock_infer(discrepancy: str) -> dict:
    """The only fake part. Replace with a real call into
    ``aircraft_nlp.models`` and keep the return shape identical.
    """
    labels = _load_labels()
    h = _stable_hash(discrepancy or "empty")
    condition = labels[h % len(labels)]
    # confidence in [0.40, 0.99], stable per input
    confidence = round(0.40 + (h % 600) / 1000.0, 3)
    keyword = _pick_keyword(discrepancy or "")
    explanation = (
        f'The phrase "{keyword}" in the discrepancy is the strongest '
        f'signal for a "{condition}" condition.'
    )
    return {
        "predicted_condition": condition,
        "confidence": confidence,
        "xai": {"keyword": keyword, "explanation": explanation},
        "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
    }


def run_prediction(rows: list[dict]) -> dict:
    """rows: list of dicts; each should contain a 'Discrepancy' field
    (case-insensitive fallbacks supported)."""
    predictions = []
    for i, row in enumerate(rows or []):
        discrepancy = (
            row.get("Discrepancy")
            or row.get("discrepancy")
            or row.get("description")
            or ""
        )
        result = _mock_infer(str(discrepancy))
        predictions.append(
            {
                "row_id": row.get("id", i),
                "discrepancy": discrepancy,
                **result,
            }
        )
    return {
        "model_version": MODEL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
    }
