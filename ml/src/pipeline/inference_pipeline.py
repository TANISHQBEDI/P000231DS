from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.components.confidence_router import ConfidenceRouter
from src.components.feature_builder import FeatureBuilder
from src.components.level1_classifier import Level1Classifier
from src.components.level2_classifier import Level2Classifier
from src.components.text_cleaner import TextCleaner
from src.components.xai import TfidfExplainer

logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, artifacts_dir: str, mapping_path: str) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.mapping_path = mapping_path
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        vectorizer_path = self.artifacts_dir / "tfidf_vectorizer.joblib"
        level1_path = self.artifacts_dir / "level1_classifier.joblib"
        level2_path = self.artifacts_dir / "level2_classifiers.joblib"
        metadata_path = self.artifacts_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.cleaned_column = metadata.get("cleaned_column", "discrepancy_text_clean")
        self.text_column = metadata.get("text_column", "discrepancy_text")

        self.feature_builder = FeatureBuilder.load(str(vectorizer_path))
        self.level1_model = Level1Classifier.load(str(level1_path))
        self.level2_model = Level2Classifier.load(str(level2_path))

        self.text_cleaner = TextCleaner()
        self.explainer = TfidfExplainer(self.feature_builder.vectorizer)
        self.router = ConfidenceRouter()

    def predict(self, discrepancy_text: str) -> dict[str, Any]:
        data = {self.text_column: discrepancy_text}
        cleaned = self.text_cleaner.clean(
            df=self._single_row(data),
            text_column=self.text_column,
            output_column=self.cleaned_column,
        )

        cleaned_text = cleaned[self.cleaned_column].iloc[0]
        features = self.feature_builder.transform([cleaned_text])

        level1_pred = self.level1_model.predict(features)[0]
        level1_proba = self.level1_model.predict_proba(features)[0]
        level1_conf = float(level1_proba.max())

        level2_pred, level2_confs = self.level2_model.predict_with_confidence(
            features,
            [level1_pred],
        )
        level2_conf = float(level2_confs[0])

        combined_conf = min(level1_conf, level2_conf)
        route = self.router.route(combined_conf)

        explanation = self.explainer.explain(cleaned_text, self.level2_model.models.get(level1_pred))

        return {
            "input": discrepancy_text,
            "cleaned_text": cleaned_text,
            "level1": level1_pred,
            "level2": level2_pred[0],
            "confidence": combined_conf,
            "route": route.decision,
            "explanation": explanation,
        }

    @staticmethod
    def _single_row(payload: dict[str, str]):
        import pandas as pd

        return pd.DataFrame([payload])
