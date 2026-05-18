# predict.py - runs UMEC inference using configs/core and saved models.

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from umec.data.preprocessing import normalize_tokens, preprocess_dataframe
from umec.data.resources import load_failure_keywords, load_token_mappings
from umec.data.validation import validate_columns
from umec.utils.config import load_config
from umec.utils.serialization import load_model

predict_bp = Blueprint("predict", __name__)

_CACHE: dict[str, dict[str, object]] = {}
_XAI_TOP_K = 3


def _get_config_dir() -> Path:
    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _load_assets() -> dict[str, object]:
    config_dir = _get_config_dir().resolve()
    cache_key = str(config_dir)
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    cfg = load_config(config_dir)
    umec = load_model(Path(cfg.project.model_dir) / "umec.joblib")
    token_map = None
    if cfg.data.preprocess.get("enabled", True):
        token_map = load_token_mappings(cfg.data.resources["token_mappings"])

    token_clf = None
    keyword_index = None
    try:
        token_clf = load_model(Path(cfg.project.model_dir) / "token_matching.joblib")
        failure_keywords = load_failure_keywords(cfg.data.resources["failure_keywords"])
        token_vocab = token_clf.vectorizer.vocabulary_
        keyword_index = {}
        for label, keywords in failure_keywords.items():
            idxs = []
            for kw in keywords:
                norm_kw = normalize_tokens(str(kw).lower().strip(), token_map)
                if norm_kw in token_vocab:
                    idxs.append((kw, token_vocab[norm_kw]))
            keyword_index[str(label)] = idxs
    except Exception:
        token_clf = None
        keyword_index = None

    cached = {
        "cfg": cfg,
        "umec": umec,
        "token_map": token_map,
        "token_clf": token_clf,
        "keyword_index": keyword_index,
    }
    _CACHE[cache_key] = cached
    return cached


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Body must include a non-empty 'rows' list."}), 400
    try:
        assets = _load_assets()
        cfg = assets["cfg"]
        umec = assets["umec"]
        token_map = assets["token_map"]
        token_clf = assets["token_clf"]
        keyword_index = assets["keyword_index"]

        df = pd.DataFrame(rows)
        source_text = cfg.data.source_text_column
        if source_text is None and cfg.data.text_column.startswith("processed_"):
            source_text = cfg.data.text_column.replace("processed_", "", 1)
        source_text = source_text or cfg.data.text_column

        validate_columns(df, [source_text])

        if cfg.data.preprocess.get("enabled", True):
            df = preprocess_dataframe(
                df,
                text_column=source_text,
                output_column=cfg.data.text_column,
                preprocess_cfg=cfg.data.preprocess,
                token_map=token_map,
            )

        preds, reduction = umec.predict(df, column_name=cfg.data.text_column)
        class_scores = umec.class_score_df(reduction)

        score_vals = class_scores.values
        score_vals = score_vals - score_vals.max(axis=1, keepdims=True)
        exp_scores = np.exp(score_vals)
        prob_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        tfidf_matrix = None
        if token_clf is not None and keyword_index:
            tfidf_matrix = token_clf.vectorizer.transform(
                df[cfg.data.text_column].fillna("").astype(str)
            )

        predictions = []
        for idx, label in preds.items():
            row = df.loc[idx]
            row_id = row.get("id", idx)
            discrepancy = row.get(source_text, "")

            row_probs = prob_scores[class_scores.index.get_loc(idx)]
            top1 = float(np.max(row_probs)) if row_probs.size else 0.0
            top2 = float(np.partition(row_probs, -2)[-2]) if row_probs.size > 1 else 0.0
            confidence = top1 / max(top1 + top2, 1e-9)

            xai_keyword = "n/a"
            xai_expl = "n/a"
            if tfidf_matrix is not None and keyword_index:
                kw_idxs = keyword_index.get(str(label), [])
                if kw_idxs:
                    row_vec = tfidf_matrix[df.index.get_loc(idx)]
                    weights = []
                    for kw, kw_idx in kw_idxs:
                        val = row_vec[0, kw_idx]
                        if val > 0:
                            weights.append((kw, float(val)))
                    if weights:
                        weights.sort(key=lambda x: x[1], reverse=True)
                        top = weights[:_XAI_TOP_K]
                        total_w = sum(w for _, w in top) or 1.0
                        formatted = "|".join(
                            f"{kw}={w / total_w * 100.0:.1f}%" for kw, w in top
                        )
                        xai_keyword = str(top[0][0])
                        xai_expl = formatted

            try:
                row_id_val = int(row_id)
            except (TypeError, ValueError):
                row_id_val = int(idx)

            predictions.append(
                {
                    "row_id": row_id_val,
                    "discrepancy": str(discrepancy),
                    "predicted_condition": str(label),
                    "confidence": float(confidence),
                    "xai": {"keyword": xai_keyword, "explanation": xai_expl},
                    "low_confidence": confidence < 0.60,
                }
            )

        return jsonify(
            {
                "model_version": f"umec-{Path(cfg.project.model_dir).name}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "predictions": predictions,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
