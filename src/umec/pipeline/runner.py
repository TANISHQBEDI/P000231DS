from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from umec.data.io import read_data, save_data
from umec.data.preprocessing import preprocess_dataframe, apply_label_map
from umec.data.resources import load_failure_keywords, load_label_mappings, load_token_mappings
from umec.data.validation import validate_columns
from umec.evaluation.metrics import classification_report_df, macro_f1, top_k_accuracy
from umec.evaluation.plots import plot_confusion_matrix
from umec.models.semantic_similarity import SemanticSimilarityClassifier, SemanticSimilarityConfig
from umec.models.token_matching import TokenMatchingClassifier, TokenMatchingConfig
from umec.models.umec import UMECClassifier, UMECConfig
from umec.utils.config import load_config
from umec.utils.logging import get_logger
from umec.utils.paths import ensure_dir
from umec.utils.serialization import save_model
from umec.utils.seed import set_seed


def _init_classifiers(cfg) -> Tuple[TokenMatchingClassifier, SemanticSimilarityClassifier]:
    token_params = dict(cfg.models.token_matching)
    if "ngram_range" in token_params and isinstance(token_params["ngram_range"], list):
        token_params["ngram_range"] = tuple(token_params["ngram_range"])
    token_cfg = TokenMatchingConfig(**token_params)
    sem_cfg = SemanticSimilarityConfig(**cfg.models.semantic_similarity)

    failure_keywords = load_failure_keywords(cfg.data.resources["failure_keywords"])
    token_map = load_token_mappings(cfg.data.resources["token_mappings"])

    token_clf = TokenMatchingClassifier(
        failure_keywords=failure_keywords,
        token_map=token_map,
        config=token_cfg,
    )

    semantic_clf = SemanticSimilarityClassifier(
        failure_keywords=failure_keywords,
        config=sem_cfg,
    )

    return token_clf, semantic_clf


def _load_and_preprocess(cfg) -> pd.DataFrame:
    read_kwargs = cfg.data.read_kwargs or {}
    df = read_data(cfg.data.path, file_format=cfg.data.format, **read_kwargs)

    source_text = cfg.data.source_text_column
    if source_text is None and cfg.data.text_column.startswith("processed_"):
        source_text = cfg.data.text_column.replace("processed_", "", 1)
    source_text = source_text or cfg.data.text_column

    source_label = cfg.data.source_label_column
    if source_label is None and cfg.data.label_column.startswith("processed_"):
        source_label = cfg.data.label_column.replace("processed_", "", 1)

    source_required = [source_text]
    if source_label:
        source_required.append(source_label)
    validate_columns(df, source_required)

    token_map = load_token_mappings(cfg.data.resources["token_mappings"])
    label_map = load_label_mappings(cfg.data.resources["label_mappings"])

    if cfg.data.preprocess.get("enabled", True):
        df = preprocess_dataframe(
            df,
            text_column=source_text,
            output_column=cfg.data.text_column,
            preprocess_cfg=cfg.data.preprocess,
            token_map=token_map,
        )

    if cfg.data.label_column in df.columns:
        df[cfg.data.label_column] = apply_label_map(df[cfg.data.label_column].astype(str), label_map)
    elif source_label and source_label in df.columns:
        df[cfg.data.label_column] = apply_label_map(df[source_label].astype(str), label_map)

    output_path = cfg.data.output.get("processed_path")
    if output_path:
        save_data(df, output_path, file_format=Path(output_path).suffix.lstrip("."))

    if cfg.data.required_columns:
        validate_columns(df, cfg.data.required_columns)

    return df


def run_train(config_dir: str) -> dict:
    cfg = load_config(config_dir)
    logger = get_logger("pipeline", cfg.project.log_level)
    set_seed(cfg.project.random_state)

    logger.info("Loading and preprocessing data")
    df = _load_and_preprocess(cfg)
    logger.info("Data ready: %s rows", len(df))
    token_clf, semantic_clf = _init_classifiers(cfg)

    logger.info("Fitting token matching classifier")
    token_clf.fit(df[cfg.data.text_column].fillna("").astype(str))

    logger.info("Fitting semantic similarity classifier")
    semantic_clf.fit(df[cfg.data.text_column].fillna("").astype(str))

    umec_cfg = UMECConfig(
        ecoc_scheme=cfg.models.umec["ecoc"]["scheme"],
        aggregation=cfg.models.umec["aggregation"],
        prior_weight=cfg.models.umec["decode"]["prior_weight"],
        allow_unclassified=cfg.models.umec["decode"]["allow_unclassified"],
        unclassified_threshold=cfg.models.umec["decode"]["unclassified_threshold"],
    )

    umec = UMECClassifier(
        classifiers=[semantic_clf, token_clf],
        config=umec_cfg,
    )

    y = df[cfg.data.label_column] if cfg.data.label_column in df.columns else None
    logger.info("Fitting UMEC ensemble")
    umec.fit(df, y=y, column_name=cfg.data.text_column)

    logger.info("UMEC training complete")

    sample_n = min(5, len(df))
    if sample_n > 0:
        sample_df = df.head(sample_n)
        tm_preds, tm_scores = token_clf.predict(sample_df, column_name=cfg.data.text_column)
        ss_preds, ss_scores = semantic_clf.predict(sample_df, column_name=cfg.data.text_column)
        umec_preds, _ = umec.predict(sample_df, column_name=cfg.data.text_column)

        logger.info("Sample predictions (first %s rows):", sample_n)
        for i, idx in enumerate(sample_df.index):
            tm_top = tm_scores.loc[idx].sort_values(ascending=False).head(3)
            ss_top = ss_scores.loc[idx].sort_values(ascending=False).head(3)
            logger.info(
                "Row %s | token=%s | semantic=%s | umec=%s",
                idx,
                tm_top.to_dict(),
                ss_top.to_dict(),
                umec_preds.loc[idx],
            )

    model_dir = ensure_dir(cfg.project.model_dir)
    save_model(token_clf, Path(model_dir) / "token_matching.joblib")
    save_model(semantic_clf, Path(model_dir) / "semantic_similarity.joblib")
    save_model(umec, Path(model_dir) / "umec.joblib")
    logger.info("Saved models to %s", model_dir)

    return {
        "config": cfg,
        "data": df,
        "token_clf": token_clf,
        "semantic_clf": semantic_clf,
        "umec": umec,
    }


def run_evaluate(config_dir: str) -> dict:
    artifacts = run_train(config_dir)
    cfg = artifacts["config"]
    df = artifacts["data"]
    umec = artifacts["umec"]

    y_true = df[cfg.data.label_column] if cfg.data.label_column in df.columns else None
    if y_true is None:
        raise ValueError("Label column not found; evaluation requires labels.")

    y_pred, reduction = umec.predict(df, column_name=cfg.data.text_column)
    class_scores = umec.class_score_df(reduction)

    labels = list(umec.classes) + ["unclassified"]
    report_df = classification_report_df(y_true, y_pred, labels=labels)
    macro = macro_f1(y_true, y_pred, labels=labels)

    metrics_dir = ensure_dir(Path(cfg.project.output_dir) / "metrics")
    report_path = metrics_dir / "umec_classification_report.csv"
    report_df.to_csv(report_path)

    fig_dir = ensure_dir(Path(cfg.project.output_dir) / "figures")
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        title="UMEC Confusion Matrix",
        save_path=str(fig_dir / "umec_confusion_matrix.png"),
    )

    top2 = top_k_accuracy(class_scores, y_true, k=2)

    return {
        "report_path": str(report_path),
        "macro_f1": macro,
        "top2_accuracy": top2,
    }
