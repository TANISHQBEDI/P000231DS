"""
Tokenizer module entry point for BERT data.

This file currently imports the dataset and prepares untokenized BERT-ready
texts so a tokenizer can be added on top of it later.
"""

from __future__ import annotations

import pandas as pd

from src.bert.data_prep import prepare_bert_data
from src.ingestion.ingest import ingest_data
from src.utils.paths import RAW_FILE


def load_dataset(file_path: str = str(RAW_FILE)) -> pd.DataFrame:
    """
    Load the raw dataset used by the BERT pipeline.
    """
    return ingest_data(file_path)


def load_bert_tokenizer_inputs(
    file_path: str = str(RAW_FILE),
    text_column: str = "discrepancy",
    label_column: str = "partcondition",
    context_column: str = "part_name",
) -> tuple[list[str], list[int], dict]:
    """
    Load the dataset and return untokenized BERT inputs.
    """
    df = load_dataset(file_path)
    return prepare_bert_data(
        df=df,
        text_column=text_column,
        label_column=label_column,
        context_column=context_column,
    )


if __name__ == "__main__":
    texts, labels, metadata = load_bert_tokenizer_inputs()
    print(f"Loaded {len(texts)} samples for BERT tokenization.")
    print("Label mapping:", metadata["label_mapping"])
    print("Sample text:", texts[0] if texts else "No samples found.")
