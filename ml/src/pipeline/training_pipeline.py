from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

    level1_counts = Counter(level1_labels)
    if len(level1_counts) < 2:
        raise ValueError(
            "Level1 training requires at least 2 classes after mapping. "
            f"Found counts: {dict(level1_counts)}"
        )

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

    train_level2_counts = {}
    for level1 in sorted(set(train_l1)):
        subset = [label for label, group in zip(train_l2, train_l1) if group == level1]
        train_level2_counts[level1] = dict(Counter(subset))
        if len(train_level2_counts[level1]) < 2:
            logger.warning(
                "Level2 training has a single class for level1=%s: %s",
                level1,
                train_level2_counts[level1],
            )

    pred_l1 = level1_model.predict(test_features)
    pred_l2, _ = level2_model.predict_with_confidence(test_features, pred_l1)

    level1_labels_sorted = sorted(set(level1_labels))
    level2_labels_sorted = sorted(set(level2_labels))

    eval_metrics = {
        "level1": {
            "accuracy": float(accuracy_score(test_l1, pred_l1)),
            "f1_weighted": float(f1_score(test_l1, pred_l1, average="weighted")),
            "classification_report": classification_report(
                test_l1,
                pred_l1,
                labels=level1_labels_sorted,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": {
                "labels": level1_labels_sorted,
                "matrix": confusion_matrix(
                    test_l1,
                    pred_l1,
                    labels=level1_labels_sorted,
                ).tolist(),
            },
        },
        "level2": {
            "accuracy": float(accuracy_score(test_l2, pred_l2)),
            "f1_weighted": float(f1_score(test_l2, pred_l2, average="weighted")),
            "classification_report": classification_report(
                test_l2,
                pred_l2,
                labels=level2_labels_sorted,
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": {
                "labels": level2_labels_sorted,
                "matrix": confusion_matrix(
                    test_l2,
                    pred_l2,
                    labels=level2_labels_sorted,
                ).tolist(),
            },
        },
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    level1_path = output_dir / "level1_classifier.joblib"
    level2_path = output_dir / "level2_classifiers.joblib"
    metadata_path = output_dir / "metadata.json"
    eval_path = output_dir / "eval.json"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    _save_confusion_matrix_plot(
        eval_metrics["level1"]["confusion_matrix"],
        plots_dir / "level1_confusion_matrix.png",
        title="Level 1 Confusion Matrix",
    )
    _save_confusion_matrix_plot(
        eval_metrics["level2"]["confusion_matrix"],
        plots_dir / "level2_confusion_matrix.png",
        title="Level 2 Confusion Matrix",
    )
    _save_f1_bar_plot(
        eval_metrics["level1"]["classification_report"],
        plots_dir / "level1_f1.png",
        title="Level 1 Per-Class F1",
    )
    _save_f1_bar_plot(
        eval_metrics["level2"]["classification_report"],
        plots_dir / "level2_f1.png",
        title="Level 2 Per-Class F1",
    )
    _save_classification_report(
        eval_metrics["level1"]["classification_report"],
        plots_dir / "level1_classification_report.txt",
        title="Level 1 Classification Report",
    )
    _save_classification_report(
        eval_metrics["level2"]["classification_report"],
        plots_dir / "level2_classification_report.txt",
        title="Level 2 Classification Report",
    )

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


def _save_confusion_matrix_plot(confusion: dict, path: Path, title: str) -> None:
    labels = confusion["labels"]
    matrix = confusion["matrix"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_f1_bar_plot(report: dict, path: Path, title: str) -> None:
    labels = [
        label
        for label in report.keys()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]
    f1_scores = [report[label]["f1-score"] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, f1_scores, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_classification_report(report: dict, path: Path, title: str) -> None:
    labels = [
        label
        for label in report.keys()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]

    lines = [title, "", "label\tprecision\trecall\tf1-score\tsupport"]
    for label in labels:
        metrics = report[label]
        lines.append(
            f"{label}\t"
            f"{metrics['precision']:.3f}\t"
            f"{metrics['recall']:.3f}\t"
            f"{metrics['f1-score']:.3f}\t"
            f"{int(metrics['support'])}"
        )

    for avg_label in ("macro avg", "weighted avg"):
        metrics = report.get(avg_label, {})
        if metrics:
            lines.append(
                f"{avg_label}\t"
                f"{metrics['precision']:.3f}\t"
                f"{metrics['recall']:.3f}\t"
                f"{metrics['f1-score']:.3f}\t"
                f"{int(metrics['support'])}"
            )

    accuracy = report.get("accuracy")
    if accuracy is not None:
        lines.append(f"accuracy\t{accuracy:.3f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
