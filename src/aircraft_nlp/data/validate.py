# src/data/validation.py

import pandas as pd

REQUIRED_COLUMNS = ["Discrepancy", "PartCondition"]

def validate_raw(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Input dataframe is empty")


def validate_processed(df: pd.DataFrame):
    if "text" not in df or "label" not in df:
        raise ValueError("Processed dataframe must contain 'text' and 'label'")

    if df["text"].isnull().any():
        raise ValueError("Null values in text column")