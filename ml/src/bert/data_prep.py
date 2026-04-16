"""
Prepare raw text data for BERT classification without tokenization.

This module:
- validates the required text and label columns
- fills and normalizes missing values
- optionally combines `part_name` with the main text column for richer context
- encodes string labels into integer ids
- creates train/validation/test splits using raw text strings

Tokenization is intentionally not handled here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class BERTDatasetSplits:
    """Container for untokenized BERT dataset splits."""

    train_texts: list[str]
    val_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    val_labels: list[int]
    test_labels: list[int]
    label_encoder: LabelEncoder
    label_mapping: dict[str, int]


class BERTDataPreparer:
    """
    Prepare raw text and labels for a downstream BERT tokenizer/model.

    The output remains untokenized so a separate tokenizer step can be used
    later in training or inference.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "discrepancy",
        label_column: str = "partcondition",
        context_column: str = "part_name",
    ) -> None:
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.context_column = context_column
        self.label_encoder = LabelEncoder()
        self.prepared_df: pd.DataFrame | None = None

    def _validate_columns(self) -> None:
        if self.df.empty:
            raise ValueError("Input dataframe is empty.")

        missing_columns = [
            column
            for column in (self.text_column, self.label_column)
            if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @staticmethod
    def _normalize_text(series: pd.Series) -> pd.Series:
        return (
            series.fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    def prepare_dataframe(self) -> pd.DataFrame:
        """
        Return a cleaned dataframe with BERT-ready raw text and encoded labels.
        """
        self._validate_columns()

        prepared = self.df.copy()
        prepared[self.text_column] = self._normalize_text(prepared[self.text_column])
        prepared[self.label_column] = (
            prepared[self.label_column].fillna("unknown").astype(str).str.strip()
        )

        if self.context_column in prepared.columns:
            prepared[self.context_column] = self._normalize_text(prepared[self.context_column])
            prepared["bert_text"] = prepared.apply(
                lambda row: (
                    f"Part: {row[self.context_column]} | Issue: {row[self.text_column]}"
                    if row[self.context_column]
                    else row[self.text_column]
                ),
                axis=1,
            )
        else:
            prepared["bert_text"] = prepared[self.text_column]

        prepared = prepared.loc[
            (prepared["bert_text"] != "") & (prepared[self.label_column] != "")
        ].copy()

        if prepared.empty:
            raise ValueError("No valid rows remain after preparing the dataset.")

        prepared["label_id"] = self.label_encoder.fit_transform(prepared[self.label_column])
        self.prepared_df = prepared.reset_index(drop=True)
        return self.prepared_df

    def get_texts_and_labels(self) -> tuple[list[str], list[int]]:
        """
        Return untokenized texts and encoded labels for the full dataset.
        """
        if self.prepared_df is None:
            self.prepare_dataframe()

        assert self.prepared_df is not None
        texts = self.prepared_df["bert_text"].tolist()
        labels = self.prepared_df["label_id"].tolist()
        return texts, labels

    def get_label_mapping(self) -> dict[str, int]:
        """
        Return the mapping from original label to encoded integer id.
        """
        if self.prepared_df is None:
            self.prepare_dataframe()

        return {
            label: int(index)
            for index, label in enumerate(self.label_encoder.classes_)
        }

    def create_splits(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
    ) -> BERTDatasetSplits:
        """
        Create train/validation/test splits from untokenized BERT inputs.

        `val_size` is the fraction of the full dataset reserved for validation.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 <= val_size < 1:
            raise ValueError("val_size must be between 0 and 1.")
        if test_size + val_size >= 1:
            raise ValueError("test_size + val_size must be less than 1.")

        texts, labels = self.get_texts_and_labels()

        label_series: list[int] | None = labels if stratify and len(set(labels)) > 1 else None
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=label_series,
        )

        if val_size == 0:
            return BERTDatasetSplits(
                train_texts=x_train_val,
                val_texts=[],
                test_texts=x_test,
                train_labels=y_train_val,
                val_labels=[],
                test_labels=y_test,
                label_encoder=self.label_encoder,
                label_mapping=self.get_label_mapping(),
            )

        relative_val_size = val_size / (1 - test_size)
        val_stratify: list[int] | None = (
            y_train_val if stratify and len(set(y_train_val)) > 1 else None
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=random_state,
            stratify=val_stratify,
        )

        return BERTDatasetSplits(
            train_texts=x_train,
            val_texts=x_val,
            test_texts=x_test,
            train_labels=y_train,
            val_labels=y_val,
            test_labels=y_test,
            label_encoder=self.label_encoder,
            label_mapping=self.get_label_mapping(),
        )


def prepare_bert_data(
    df: pd.DataFrame,
    text_column: str = "discrepancy",
    label_column: str = "partcondition",
    context_column: str = "part_name",
) -> tuple[list[str], list[int], dict[str, Any]]:
    """
    Prepare untokenized text and encoded labels for BERT.

    Returns:
        texts: raw text strings ready for tokenization later
        labels: encoded integer labels
        metadata: prepared dataframe and label metadata
    """
    preparer = BERTDataPreparer(
        df=df,
        text_column=text_column,
        label_column=label_column,
        context_column=context_column,
    )
    prepared_df = preparer.prepare_dataframe()
    texts, labels = preparer.get_texts_and_labels()

    metadata = {
        "prepared_df": prepared_df,
        "label_encoder": preparer.label_encoder,
        "label_mapping": preparer.get_label_mapping(),
    }
    return texts, labels, metadata


def prepare_bert_splits(
    df: pd.DataFrame,
    text_column: str = "discrepancy",
    label_column: str = "partcondition",
    context_column: str = "part_name",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> BERTDatasetSplits:
    """
    Prepare untokenized BERT data and split it into train/val/test sets.
    """
    preparer = BERTDataPreparer(
        df=df,
        text_column=text_column,
        label_column=label_column,
        context_column=context_column,
    )
    preparer.prepare_dataframe()
    return preparer.create_splits(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )
