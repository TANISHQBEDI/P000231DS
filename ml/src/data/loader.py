from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.validator import DataValidator


class DataLoader:
    def __init__(self, validator: DataValidator | None = None) -> None:
        self.validator = validator or DataValidator()

    def load(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext == ".xlsx":
            df = pd.read_excel(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported file extension. Use .xlsx or .csv")

        result = self.validator.validate_and_standardize(df)
        return result.df
