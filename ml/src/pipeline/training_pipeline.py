from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.components.feature_builder import FeatureBuilder
from src.components.label_mapper import LabelMapper
from src.components.level1_classifier import Level1Classifier
from src.components.level2_classifier import Level2Classifier
from src.components.text_cleaner import TextCleaner
from src.data.loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    data_path: str
    mapping_path: str
    output_dir: str = "models/hierarchical"
    text_column: str = "discrepancy_text"
    label_column: str = "part_condition"
    cleaned_column: str = "discrepancy_text_clean"
    max_features: int = 5000
    test_size: float = 0.2
    random_state: int = 42


def train(config: TrainingConfig) -> dict[str, str]:
    loader = DataLoader()
    df = loader.load(config.data_path)

    cleaner = TextCleaner()
    df = cleaner.clean(df, text_column=config.text_column, output_column=config.cleaned_column)

    mapper = LabelMapper(config.mapping_path)
    df = mapper.map_labels(df, label_column=config.label_column)

    df = df[(df["level1"] != "UNKNOWN") & (df["level2"] != "UNKNOWN")]
    if df.empty:
        raise ValueError("No labeled data remains after mapping.")

    texts = df[config.cleaned_column].tolist()
    level1_labels = df["level1"].tolist()
    level2_labels = df["level2"].tolist()

    (train_texts, test_texts, train_l1, test_l1, train_l2, test_l2) = train_test_split(
        texts,
        level1_labels,
        level2_labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=level1_labels if len(set(level1_labels)) > 1 else None,
    )

    feature_builder = FeatureBuilder(max_features=config.max_features)
    train_features = feature_builder.fit_transform(train_texts)
    test_features = feature_builder.transform(test_texts)

    level1_model = Level1Classifier().fit(train_features, train_l1)
    level2_model = Level2Classifier().fit(train_features, train_l1, train_l2)

    pred_l1 = level1_model.predict(test_features)
    pred_l2, _ = level2_model.predict_with_confidence(test_features, pred_l1)

    eval_metrics = {
        "level1": {
            "accuracy": float(accuracy_score(test_l1, pred_l1)),
            "f1_weighted": float(f1_score(test_l1, pred_l1, average="weighted")),
        },
        "level2": {
            "accuracy": float(accuracy_score(test_l2, pred_l2)),
            "f1_weighted": float(f1_score(test_l2, pred_l2, average="weighted")),
        },
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    level1_path = output_dir / "level1_classifier.joblib"
    level2_path = output_dir / "level2_classifiers.joblib"
    metadata_path = output_dir / "metadata.json"
    eval_path = output_dir / "eval.json"

    feature_builder.save(str(vectorizer_path))
    level1_model.save(str(level1_path))
    level2_model.save(str(level2_path))

    metadata = {
        "text_column": config.text_column,
        "label_column": config.label_column,
        "cleaned_column": config.cleaned_column,
        "mapping_path": config.mapping_path,
        "max_features": config.max_features,
        "level1_labels": sorted(set(level1_labels)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    logger.info("Training complete. Artifacts saved to %s", output_dir)
    logger.info(
        "Eval metrics | L1 acc: %.3f, L1 f1: %.3f | L2 acc: %.3f, L2 f1: %.3f",
        eval_metrics["level1"]["accuracy"],
        eval_metrics["level1"]["f1_weighted"],
        eval_metrics["level2"]["accuracy"],
        eval_metrics["level2"]["f1_weighted"],
    )

    return {
        "vectorizer": str(vectorizer_path),
        "level1_model": str(level1_path),
        "level2_model": str(level2_path),
        "metadata": str(metadata_path),
        "eval": str(eval_path),
    }
