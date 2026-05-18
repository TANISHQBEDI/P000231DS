from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List

import numpy as np
import pandas as pd

from umec.models.base import BaseUnsupervisedClassifier


@dataclass
class UMECConfig:
    ecoc_scheme: str = "pairwise"
    aggregation: str = "mean"
    prior_weight: float = 0.35
    allow_unclassified: bool = True
    unclassified_threshold: float = 0.0


class UMECClassifier:
    """UMEC ensemble using ECOC-like reduction with max-based order statistics."""

    def __init__(
        self,
        classifiers: List[BaseUnsupervisedClassifier],
        classes: list[str] | None = None,
        ecoc_matrix: np.ndarray | None = None,
        config: UMECConfig | None = None,
    ) -> None:
        self.classifiers = classifiers
        self.classes = classes
        self.ecoc_matrix = ecoc_matrix
        self.config = config or UMECConfig()

        self.bit_labels: list[str] = []
        self.class_priors: pd.Series | None = None

    def _score_df(self, clf: BaseUnsupervisedClassifier, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        out = clf.predict(df, column_name=column_name)
        scores = out[1] if isinstance(out, tuple) else out
        if not isinstance(scores, pd.DataFrame):
            scores = pd.DataFrame(scores, index=df.index, columns=clf.classes)
        return scores

    def _common_classes(self, score_dfs: list[pd.DataFrame]) -> list[str]:
        common = set(score_dfs[0].columns)
        for s in score_dfs[1:]:
            common &= set(s.columns)
        if not common:
            raise ValueError("No shared class columns across classifiers.")
        return sorted(common)

    def _build_pairwise_ecoc(self, classes: list[str]) -> np.ndarray:
        bits = []
        labels = []
        for i, j in combinations(range(len(classes)), 2):
            col = np.zeros(len(classes))
            col[i] = 1
            col[j] = -1
            bits.append(col)
            labels.append(f"{classes[i]}_vs_{classes[j]}")
        self.bit_labels = labels
        return np.stack(bits, axis=1)

    def _resolve_ecoc(self, classes: list[str]) -> np.ndarray:
        if self.ecoc_matrix is not None:
            if self.ecoc_matrix.shape[0] != len(classes):
                raise ValueError("ECOC matrix row count must match number of classes.")
            return self.ecoc_matrix

        if self.config.ecoc_scheme == "pairwise":
            return self._build_pairwise_ecoc(classes)

        raise ValueError(f"Unsupported ECOC scheme: {self.config.ecoc_scheme}")

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None, column_name: str = "processed_discrepancy") -> "UMECClassifier":
        score_dfs = [self._score_df(clf, df, column_name) for clf in self.classifiers]
        common_classes = self._common_classes(score_dfs)
        self.classes = self.classes or common_classes

        if y is not None:
            y = y.astype(str)
            self.class_priors = y.value_counts(normalize=True).reindex(self.classes).fillna(1e-6)
        else:
            self.class_priors = pd.Series(1.0 / len(self.classes), index=self.classes)

        self.ecoc_matrix = self._resolve_ecoc(self.classes)
        return self

    def _reduction_stats(self, score_df: pd.DataFrame) -> np.ndarray:
        if self.ecoc_matrix is None:
            raise ValueError("ECOC matrix is not initialized. Call fit() first.")

        scores = score_df[self.classes].values
        stats = []
        for bit_idx in range(self.ecoc_matrix.shape[1]):
            code = self.ecoc_matrix[:, bit_idx]
            pos_idx = np.where(code == 1)[0]
            neg_idx = np.where(code == -1)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                stat = np.zeros(scores.shape[0])
            else:
                pos_max = scores[:, pos_idx].max(axis=1)
                neg_max = scores[:, neg_idx].max(axis=1)
                stat = pos_max - neg_max
            stats.append(stat)

        return np.stack(stats, axis=1)

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        score_dfs = [self._score_df(clf, df, column_name) for clf in self.classifiers]
        aligned_scores = [s[self.classes] for s in score_dfs]
        reductions = [self._reduction_stats(s) for s in aligned_scores]

        if self.config.aggregation == "sum":
            agg = np.sum(reductions, axis=0)
        else:
            agg = np.mean(reductions, axis=0)

        bit_labels = self.bit_labels or [f"bit_{i}" for i in range(agg.shape[1])]
        return pd.DataFrame(agg, columns=bit_labels, index=df.index)

    def class_score_df(self, reduction_df: pd.DataFrame) -> pd.DataFrame:
        if self.ecoc_matrix is None:
            raise ValueError("ECOC matrix is not initialized. Call fit() first.")

        reduction = reduction_df.values
        code = self.ecoc_matrix
        denom = (code != 0).sum(axis=1)
        denom = np.where(denom == 0, 1, denom)
        margins = (reduction @ code.T) / denom

        priors = self.class_priors.reindex(self.classes).fillna(1e-6).values
        prior_adj = self.config.prior_weight * np.log(priors + 1e-9)
        scores = margins + prior_adj

        return pd.DataFrame(scores, columns=self.classes, index=reduction_df.index)

    def _decode(self, class_scores: pd.Series) -> tuple[str, float]:
        best_label = class_scores.idxmax()
        best_score = float(class_scores.max())
        return best_label, best_score

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        reduction_df = self.transform(df, column_name)
        class_scores = self.class_score_df(reduction_df)

        labels = []
        for _, row in class_scores.iterrows():
            label, score = self._decode(row)
            if self.config.allow_unclassified and score < self.config.unclassified_threshold:
                labels.append("unclassified")
            else:
                labels.append(label)

        return pd.Series(labels, index=df.index), reduction_df
