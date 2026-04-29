from __future__ import annotations

from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureBuilder:
    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
        )

    def fit(self, texts: Iterable[str]) -> "FeatureBuilder":
        self.vectorizer.fit(list(texts))
        return self

    def transform(self, texts: Iterable[str]):
        return self.vectorizer.transform(list(texts))

    def fit_transform(self, texts: Iterable[str]):
        return self.vectorizer.fit_transform(list(texts))

    def get_feature_names(self) -> list[str]:
        return list(self.vectorizer.get_feature_names_out())

    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)

    @staticmethod
    def load(path: str) -> "FeatureBuilder":
        builder = FeatureBuilder()
        builder.vectorizer = joblib.load(path)
        return builder
