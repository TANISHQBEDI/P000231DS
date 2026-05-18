from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_data(path: str | Path, file_format: str | None = None, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        df = pd.read_csv(path, **kwargs)
    elif fmt in {"xlsx", "xls"}:
        df = pd.read_excel(path, **kwargs)
    elif fmt == "parquet":
        df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {fmt}")

    return df


def save_data(df: pd.DataFrame, path: str | Path, file_format: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
