from __future__ import annotations

from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression


class Level1Classifier:
    def __init__(self, model: Any | None = None) -> None:
        self.model = model or LogisticRegression(max_iter=1000, multi_class="auto")

    def fit(self, features, labels: list[str]) -> "Level1Classifier":
        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2:
            raise ValueError(
                "Level1 classifier requires at least 2 classes. "
                f"Found: {unique_labels}"
            )
        self.model.fit(features, labels)
        return self

    def predict(self, features) -> list[str]:
        return self.model.predict(features).tolist()

    def predict_proba(self, features):
        return self.model.predict_proba(features)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str) -> "Level1Classifier":
        classifier = Level1Classifier()
        classifier.model = joblib.load(path)
        return classifier
