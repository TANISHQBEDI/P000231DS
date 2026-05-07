# ==================================
# ModernBERT Tokenization Module
# ==================================

import re
import pandas as pd
from transformers import AutoTokenizer


def _clean_text(text: str) -> str:
    """Minimal clean before tokenization: strip and collapse whitespace."""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


class ModernBERTTokenizer:
    """
    Wrapper around HuggingFace AutoTokenizer for ModernBERT.
    Handles batched tokenization with padding, truncation, and attention masks.
    Keeps both fine-tuning (return_tensors='pt') and
    feature extraction (return_tensors='np') paths open.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts: list[str], return_tensors: str = "pt") -> dict:
        """
        Tokenize a list of texts.

        Parameters:
            texts: list of raw strings
            return_tensors: 'pt' (PyTorch), 'np' (NumPy), or None (Python lists)

        Returns dict with keys: input_ids, attention_mask
        """
        cleaned = [_clean_text(t) for t in texts]
        return self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )

    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "discrepancy",
        return_tensors: str = "pt",
    ) -> dict:
        """Convenience method: tokenize directly from a DataFrame column."""
        texts = df[text_column].fillna("").tolist()
        return self.tokenize(texts, return_tensors=return_tensors)

    def tokenize_single(self, text: str, return_tensors: str = "pt") -> dict:
        """Tokenize a single string."""
        return self.tokenize([text], return_tensors=return_tensors)

    def decode(self, token_ids) -> str:
        """Decode token IDs back to text (useful for debugging)."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


# ==================================
# Dataset test
# ==================================
if __name__ == "__main__":
    from pathlib import Path
    from src.ingestion import ingest_data
    from src.preprocessing.text_cleaning import TextCleaner

    # ── Load & clean ──────────────────────────────────────────
    DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "raw" / "NLP_Dataset_2026.xlsx"
    print(f"Loading: {DATA_FILE}")

    df = ingest_data(DATA_FILE)
    df = TextCleaner(df).remove_null().remove_duplicates().get_data()

    print(f"Rows after cleaning : {len(df)}")
    print(f"Tokenizing column   : 'discrepancy'")
    print(f"Sample text[0]      : {df['discrepancy'].iloc[0][:80]!r}")
    print()

    # ── Tokenize a small sample first ─────────────────────────
    SAMPLE_N = 5
    t = ModernBERTTokenizer(max_length=512)
    print(f"Vocab size          : {t.vocab_size}")

    sample_out = t.tokenize(df["discrepancy"].iloc[:SAMPLE_N].tolist())
    print(f"\n--- Sample ({SAMPLE_N} rows) ---")
    print(f"input_ids shape     : {sample_out['input_ids'].shape}")
    print(f"attention_mask shape: {sample_out['attention_mask'].shape}")

    for i in range(SAMPLE_N):
        token_count = int(sample_out["attention_mask"][i].sum())
        print(f"  [{i}] tokens used: {token_count}  |  text: {df['discrepancy'].iloc[i][:60]!r}")

    # ── Original texts (decoded from token IDs) ───────────────
    print(f"\n--- Original texts ---")
    for i in range(SAMPLE_N):
        decoded = t.decode(sample_out["input_ids"][i])
        print(f"\n[{i}] {decoded}")
