from __future__ import annotations

import re
from typing import Mapping

import pandas as pd


class TextCleaner:
    DEFAULT_ABBREVIATIONS: Mapping[str, str] = {
        "nff": "no fault found",
        "u/s": "unserviceable",
        "a/c": "aircraft",
        "acft": "aircraft",
        "fwd": "forward",
        "lh": "left hand",
        "rh": "right hand",
    }

    def __init__(self, abbreviations: Mapping[str, str] | None = None) -> None:
        self.abbreviations = abbreviations or self.DEFAULT_ABBREVIATIONS

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean(
        self,
        df: pd.DataFrame,
        text_column: str = "discrepancy_text",
        output_column: str = "discrepancy_text_clean",
        original_column: str = "discrepancy_text_original",
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")

        data = df.copy()
        data[original_column] = data[text_column].fillna("").astype(str)
        cleaned = data[text_column].fillna("").astype(str).map(self._normalize)

        for short, expanded in self.abbreviations.items():
            pattern = rf"(?<!\w){re.escape(short)}(?!\w)"
            cleaned = cleaned.str.replace(pattern, expanded, regex=True)

        data[output_column] = cleaned
        return data
