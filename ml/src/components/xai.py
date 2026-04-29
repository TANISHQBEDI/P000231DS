from __future__ import annotations

from typing import Any

import numpy as np


class TfidfExplainer:
    def __init__(self, vectorizer) -> None:
        self.vectorizer = vectorizer
        self.feature_names = vectorizer.get_feature_names_out()

    def explain(self, text: str, model: Any, top_k: int = 5) -> list[dict[str, float]]:
        if not text:
            return []

        vector = self.vectorizer.transform([text])
        if not hasattr(model, "coef_"):
            return []

        predicted = model.predict(vector)[0]
        class_index = int(np.where(model.classes_ == predicted)[0][0])
        weights = model.coef_[class_index]

        contribution = vector.multiply(weights)
        scores = contribution.toarray().ravel()

        if scores.size == 0:
            return []

        top_indices = scores.argsort()[::-1][:top_k]
        explanations: list[dict[str, float]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            explanations.append({"term": str(self.feature_names[idx]), "weight": float(scores[idx])})
        return explanations
