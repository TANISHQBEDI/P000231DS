from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class LabelMapper:
    def __init__(self, mapping_path: str | Path) -> None:
        self.mapping_path = Path(mapping_path)
        self.mapping = self._load_mapping()
        self._validate_mapping()

    def _load_mapping(self) -> dict[str, dict[str, str]]:
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Label mapping not found: {self.mapping_path}")
        data = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Label mapping must be a JSON object")
        return data

    def _validate_mapping(self) -> None:
        for label, payload in self.mapping.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid mapping for label: {label}")
            if "level1" not in payload or "level2" not in payload:
                raise ValueError(f"Mapping for {label} must include level1 and level2")
            # if payload.get("level1") == "OTHER" or payload.get("level2") == "OTHER":
            #     raise ValueError("'OTHER' is not allowed as a training label")

    def map_label(self, raw_label: str) -> tuple[str, str]:
        record = self.mapping.get(raw_label)
        if not record:
            return "UNKNOWN", "UNKNOWN"
        return record.get("level1", "UNKNOWN"), record.get("level2", "UNKNOWN")

    def map_labels(
        self,
        df: pd.DataFrame,
        label_column: str = "part_condition",
        level1_column: str = "level1",
        level2_column: str = "level2",
    ) -> pd.DataFrame:
        if label_column not in df.columns:
            raise ValueError(f"Column '{label_column}' not found")

        data = df.copy()
        raw_labels = data[label_column].fillna("").astype(str).str.strip()
        mapped = raw_labels.map(self.mapping.get)

        data[level1_column] = mapped.map(lambda item: item.get("level1") if item else "UNKNOWN")
        data[level2_column] = mapped.map(lambda item: item.get("level2") if item else "UNKNOWN")

        # if (data[level1_column] == "OTHER").any() or (data[level2_column] == "OTHER").any():
        #     raise ValueError("'OTHER' is not allowed as a training label")

        return data
