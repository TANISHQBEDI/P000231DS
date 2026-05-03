# ==================================
# Training Module
# Mitchell Hughes
# 16/04/2026
# Notes:
# ================================== 
# - This module is responsible for training the BERT model on the ingested data.
# - The main function is `train_model` (line XXX), which will run through the entire training process.
# - The output will be the trained model, and will also save a copy of the model for future use.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split

from src.bert.data_prep import combine_part_condition_label, prepare_bert_splits
from src.bert.model import load_model, save_checkpoint


class _BERTTextDataset(Dataset):
    """Minimal dataset wrapper around tokenized text batches."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _build_dataloader(
    texts: list[str],
    labels: list[int],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = _BERTTextDataset(texts, labels, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def _calculate_accuracy(
    model,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    correct_predictions = 0
    total_predictions = 0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    model.train()
    if total_predictions == 0:
        return 0.0
    return correct_predictions / total_predictions


def _compute_class_weights(labels: list[int], num_labels: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_labels).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_labels * counts)
    return weights.to(device)

# Training Function 
def train_model(data: pd.DataFrame):
    print("Starting")
    """
    Train the BERT model on the provided data.

    Parameters:
    data (pd.DataFrame): The cleaned and processed data to train the model on.

    Returns:
    model: The trained BERT model.
    """
    try:
        splits = prepare_bert_splits(data)
    except ValueError as exc:
        if "least populated class in y has only 1 member" not in str(exc):
            raise

        context_column = "partname" if "partname" in data.columns else "part_name"
        text_series = data["discrepancy"].fillna("").astype(str).str.strip()
        if context_column in data.columns:
            context_series = data[context_column].fillna("").astype(str).str.strip()
            texts = [
                f"Part: {context} | Issue: {text}" if context else text
                for context, text in zip(context_series, text_series)
            ]
        else:
            texts = text_series.tolist()
        labels = (
            data["partcondition"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .map(combine_part_condition_label)
            .tolist()
        )
        filtered_pairs = [
            (text, label)
            for text, label in zip(texts, labels)
            if text and label
        ]
        if len(filtered_pairs) < 2:
            raise ValueError("Not enough valid training rows remain after filtering.") from exc

        filtered_texts, filtered_labels = zip(*filtered_pairs)
        encoded_labels = pd.Series(filtered_labels).astype("category")
        label_mapping = {
            label: index
            for index, label in enumerate(encoded_labels.cat.categories)
        }
        y = encoded_labels.cat.codes.tolist()
        x_train, x_test, y_train, y_test = train_test_split(
            list(filtered_texts),
            y,
            test_size=0.2,
            random_state=42,
            stratify=None,
        )

        class _FallbackSplits:
            def __init__(self, train_texts, train_labels, label_mapping):
                self.train_texts = train_texts
                self.train_labels = train_labels
                self.val_texts = []
                self.val_labels = []
                self.test_texts = x_test
                self.test_labels = y_test
                self.label_mapping = label_mapping

        splits = _FallbackSplits(x_train, y_train, label_mapping)
    num_labels = len(splits.label_mapping)
    if num_labels < 2:
        raise ValueError("BERT training requires at least two distinct labels.")

    model_name = "bert-base-uncased"
    batch_size = 16
    max_length = 192
    epochs = 3
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_ratio = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = _build_dataloader(
        splits.train_texts,
        splits.train_labels,
        tokenizer,
        max_length,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    evaluation_name = "test"
    evaluation_texts = splits.test_texts if splits.test_texts else splits.val_texts
    evaluation_labels = splits.test_labels if splits.test_labels else splits.val_labels
    if not evaluation_texts or not evaluation_labels:
        evaluation_name = "training"
        evaluation_texts = splits.train_texts
        evaluation_labels = splits.train_labels
    evaluation_loader = None
    if evaluation_texts and evaluation_labels:
        evaluation_loader = _build_dataloader(
            evaluation_texts,
            evaluation_labels,
            tokenizer,
            max_length,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

    model = load_model(
        model_name=model_name,
        num_labels=num_labels,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    class_weights = _compute_class_weights(splits.train_labels, num_labels, device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_state_dict = None
    best_accuracy = float("-inf")
    best_metric_name = "training"
    model.train()
    for epoch_index in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        average_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch_index + 1}/{epochs} loss: {average_loss:.4f}")

        current_metric_name = evaluation_name if evaluation_loader is not None else "training"
        current_loader = evaluation_loader if evaluation_loader is not None else train_loader
        current_accuracy = _calculate_accuracy(model, current_loader, device)
        print(f"Epoch {epoch_index + 1}/{epochs} {current_metric_name} accuracy: {current_accuracy:.2%}")

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_metric_name = current_metric_name
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if evaluation_loader is not None or best_accuracy > float("-inf"):
        print(f"Best {best_metric_name} accuracy: {best_accuracy:.2%}")
    else:
        print("Accuracy: unavailable because no data was available for evaluation.")

    model_dir = Path(__file__).resolve().parent.parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, str(model_dir / "bert_model.pt"))

    return model
