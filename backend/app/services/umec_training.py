"""Training wrapper that calls the UMEC pipeline."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from umec.pipeline.runner import run_evaluate, run_train
from umec.utils.config import load_config


def _get_config_dir() -> Path:
    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _model_version() -> str:
    try:
        cfg = load_config(_get_config_dir())
        return f"umec-{Path(cfg.project.model_dir).name}"
    except Exception:
        return "umec-unknown"


def run_training(dataset_meta: dict | None = None, feedback: list | None = None) -> dict:
    config_dir = _get_config_dir()
    run_train(str(config_dir))

    metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    report = {}

    try:
        eval_out = run_evaluate(str(config_dir))
        macro = eval_out.get("macro_f1")
        if macro is not None:
            metrics = {
                "accuracy": macro,
                "precision": macro,
                "recall": macro,
                "f1": macro,
            }
        report = {"report_path": eval_out.get("report_path")}
        report["top2_accuracy"] = eval_out.get("top2_accuracy")
    except Exception:
        pass

    return {
        "model_version": _model_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_feedback_records": len(feedback or []),
        "dataset_meta": dataset_meta or {},
        "metrics": metrics,
        "report": report,
    }
