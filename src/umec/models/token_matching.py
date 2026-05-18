from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from umec.data.preprocessing import normalize_tokens


@dataclass
class TokenMatchingConfig:
    ngram_range: tuple[int, int] = (1, 2)
    lowercase: bool = True
    use_idf: bool = True
    sublinear_tf: bool = True
    normalize_tokens: bool = True


class TokenMatchingClassifier:
    """Unsupervised keyword matcher using TF-IDF over domain vocabulary."""

    def __init__(
        self,
        failure_keywords: Dict[str, list[str]],
        token_map: Dict[str, str] | None = None,
        config: TokenMatchingConfig | None = None,
    ) -> None:
        self.failure_keywords = failure_keywords
        self.token_map = token_map or {}
        self.config = config or TokenMatchingConfig()

        self.classes: list[str] = list(failure_keywords.keys())
        self.vectorizer: TfidfVectorizer | None = None
        self.mapping_matrix: csr_matrix | None = None
        self.feature_names: list[str] | None = None

    def _normalize_keyword(self, keyword: str) -> str:
        keyword = keyword.lower().strip()
        if self.config.normalize_tokens:
            keyword = normalize_tokens(keyword, self.token_map)
        return keyword

    def fit(self, corpus: Iterable[str]) -> "TokenMatchingClassifier":
        tokens = sorted(
            {
                self._normalize_keyword(t)
                for sublist in self.failure_keywords.values()
                for t in sublist
            }
        )

        if not tokens:
            raise ValueError("Failure keywords are empty; cannot build vocabulary.")

        self.vectorizer = TfidfVectorizer(
            vocabulary=tokens,
            ngram_range=self.config.ngram_range,
            lowercase=self.config.lowercase,
            use_idf=self.config.use_idf,
            sublinear_tf=self.config.sublinear_tf,
        )
        self.vectorizer.fit(corpus)
        self.feature_names = list(self.vectorizer.get_feature_names_out())

        token_to_idx = {token: idx for idx, token in enumerate(self.feature_names)}
        mapping_matrix = np.zeros((len(self.feature_names), len(self.classes)))

        for class_idx, label in enumerate(self.classes):
            for token in self.failure_keywords[label]:
                token = self._normalize_keyword(token)
                if token in token_to_idx:
                    mapping_matrix[token_to_idx[token], class_idx] = 1.0

        self.mapping_matrix = csr_matrix(mapping_matrix)
        return self

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        if self.vectorizer is None or self.mapping_matrix is None:
            raise ValueError("Classifier must be fitted before transforming.")

        texts = df[column_name].fillna("").astype(str)
        if self.config.normalize_tokens and self.token_map:
            texts = texts.apply(lambda x: normalize_tokens(x, self.token_map))

        x_mat = self.vectorizer.transform(texts)
        raw_scores = x_mat.dot(self.mapping_matrix).toarray()
        return pd.DataFrame(raw_scores, columns=self.classes, index=df.index)

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        scores = self.transform(df, column_name)
        preds = scores.idxmax(axis=1)
        return preds, scores
