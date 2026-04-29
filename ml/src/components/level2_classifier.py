from __future__ import annotations

from typing import Callable

import joblib
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


class Level2Classifier:
    def __init__(self, model_factory: Callable[[], BaseEstimator] | None = None) -> None:
        self.model_factory = model_factory or (lambda: LogisticRegression(max_iter=1000))
        self.models: dict[str, BaseEstimator] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}

    def fit(self, features, level1_labels: list[str], level2_labels: list[str]) -> "Level2Classifier":
        for level1 in sorted(set(level1_labels)):
            indices = [i for i, label in enumerate(level1_labels) if label == level1]
            if not indices:
                continue

            subset_features = features[indices]
            subset_labels = [level2_labels[i] for i in indices]

            encoder = LabelEncoder()
            encoded_labels = encoder.fit_transform(subset_labels)
            model = self.model_factory()
            model.fit(subset_features, encoded_labels)

            self.models[level1] = model
            self.label_encoders[level1] = encoder

        return self

    def predict_with_confidence(
        self,
        features,
        level1_predictions: list[str],
    ) -> tuple[list[str], list[float]]:
        predictions: list[str] = []
        confidences: list[float] = []

        for idx, level1 in enumerate(level1_predictions):
            model = self.models.get(level1)
            encoder = self.label_encoders.get(level1)

            if model is None or encoder is None:
                predictions.append("UNKNOWN")
                confidences.append(0.0)
                continue

            row = features[idx]
            probs = model.predict_proba(row)
            best_idx = int(probs.argmax(axis=1)[0])
            label = encoder.inverse_transform([best_idx])[0]
            confidence = float(probs[0, best_idx])

            predictions.append(str(label))
            confidences.append(confidence)

        return predictions, confidences

    def save(self, path: str) -> None:
        joblib.dump({"models": self.models, "label_encoders": self.label_encoders}, path)

    @staticmethod
    def load(path: str) -> "Level2Classifier":
        classifier = Level2Classifier()
        payload = joblib.load(path)
        classifier.models = payload.get("models", {})
        classifier.label_encoders = payload.get("label_encoders", {})
        return classifier
