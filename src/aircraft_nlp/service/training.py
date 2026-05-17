"""Mock training wrapper.

Contract (stable):
    run_training(dataset_meta, feedback) -> {
        "model_version": str,
        "trained_at": iso8601,
        "num_feedback_records": int,
        "metrics": {"accuracy", "precision", "recall", "f1"},
        "report": {label: {"precision","recall","f1","support"}, ...},
    }

Replace the body with a real call into ``aircraft_nlp.models.train`` /
``aircraft_nlp.models.evaluate`` keeping this shape.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .prediction import MODEL_VERSION, _load_labels


def run_training(dataset_meta: dict | None = None,
                 feedback: list | None = None) -> dict:
    feedback = feedback or []
    labels = _load_labels()

    # Deterministic-ish mock metrics.
    base = 0.82
    report = {}
    for idx, label in enumerate(labels):
        p = round(min(0.99, base + (idx % 5) * 0.02), 3)
        r = round(min(0.99, base + (idx % 4) * 0.025), 3)
        f1 = round(2 * p * r / (p + r), 3)
        report[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": 20 + idx * 3,
        }

    macro_p = round(sum(v["precision"] for v in report.values()) / len(report), 3)
    macro_r = round(sum(v["recall"] for v in report.values()) / len(report), 3)
    macro_f1 = round(sum(v["f1"] for v in report.values()) / len(report), 3)

    return {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_feedback_records": len(feedback),
        "dataset_meta": dataset_meta or {},
        "metrics": {
            "accuracy": macro_f1,
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
        },
        "report": report,
    }
