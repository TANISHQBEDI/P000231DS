from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

REQUIRED_COLUMNS = ("discrepancy_text", "part_condition")
DEFAULT_COLUMN_ALIASES = {
    "discrepancy": "discrepancy_text",
    "discrepancy_text": "discrepancy_text",
    "partcondition": "part_condition",
    "part_condition": "part_condition",
}


@dataclass(frozen=True)
class ValidationResult:
    df: pd.DataFrame
    required_columns: tuple[str, str]


class DataValidator:
    def __init__(
        self,
        required_columns: tuple[str, str] = REQUIRED_COLUMNS,
        column_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.required_columns = required_columns
        self.column_aliases = {
            self._normalize_column(key): value
            for key, value in (column_aliases or DEFAULT_COLUMN_ALIASES).items()
        }

    @staticmethod
    def _normalize_column(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    def validate_and_standardize(self, df: pd.DataFrame) -> ValidationResult:
        if df is None or df.empty:
            raise ValueError("Input dataframe is empty.")

        data = df.copy()
        normalized = {col: self._normalize_column(col) for col in data.columns}
        data.rename(columns=normalized, inplace=True)

        for source, target in self.column_aliases.items():
            if source in data.columns and target not in data.columns:
                data.rename(columns={source: target}, inplace=True)

        missing = [col for col in self.required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        data = data.dropna(subset=list(self.required_columns))
        if data.empty:
            raise ValueError("No valid rows remain after validation.")

        return ValidationResult(df=data, required_columns=self.required_columns)
