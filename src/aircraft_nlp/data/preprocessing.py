# src/data/preprocessing.py

import json
import re
import pandas as pd
from importlib import resources

# ---------------------------
# Load abbreviation dictionary
# ---------------------------

def _load_abbreviations() -> dict[str, str]:
    with resources.files("aircraft_nlp.config").joinpath("abbreviations.json").open("r", encoding="utf-8") as f:
        return json.load(f)

_ABBREVIATIONS = _load_abbreviations()

_PATTERN = (
    re.compile(r"\b(" + "|".join(map(re.escape, _ABBREVIATIONS)) + r")\b")
    if _ABBREVIATIONS else None
)

# ---------------------------
# Load abbreviation dictionary
# ---------------------------

def _load_label_mappings() -> dict[str, str]:
    with resources.files("aircraft_nlp.config").joinpath("label_mappings.json").open("r", encoding="utf-8") as f:
        return json.load(f)

_LABEL_MAPPINGS = _load_label_mappings()

# ---------------------------
# Text-level transformations
# ---------------------------

def normalize_text(text: str) -> str:
    text = text.lower()

    # Expand abbreviations
    if _PATTERN:
        text = _PATTERN.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)

    # Remove special characters (keep alphanum + space)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------
# DataFrame-level steps
# ---------------------------

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Discrepancy", "PartCondition"]].copy()


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Discrepancy", "PartCondition"])
    df = df[df["Discrepancy"].str.strip() != ""]
    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw_labels = df["PartCondition"].astype(str).str.strip().str.lower()

    mapped = raw_labels.map(_LABEL_MAPPINGS)
    df["label"] = mapped.fillna("other")

    # --- observability ---
    total = len(df)
    other_count = (df["label"] == "other").sum()

    print(f"[Label Mapping] other: {other_count}/{total} ({other_count/total:.2%})")

    # Show top unknown labels (VERY useful for improving mapping)
    unknowns = raw_labels[mapped.isna()].value_counts().head(5)
    if not unknowns.empty:
        print("\n[Top Unknown Labels]")
        print(unknowns)
    
    print("\n[Label Value Counts]")
    print(df["label"].value_counts())

    return df


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["Discrepancy"].astype(str).apply(normalize_text)
    return df


# ---------------------------
# Main entrypoint
# ---------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
        df with columns: discrepancy, partCondition

    Output:
        df with columns: text, label
    """

    df = select_columns(df)
    df = drop_invalid_rows(df)
    df = normalize_labels(df)
    df = clean_text_column(df)

    return df[["text", "label"]]