# ==================================
# Evaluation Module
# Mitchell Hughes
# 16/04/2026
# Notes:
# ==================================
# - This module is responsible for evaluating the performance of BERT models.
# - The main function is `evaluate_model`, which evaluates tokenized tensors.
# - `evaluate_dataloader` is used by training to calculate epoch metrics.
# - The output contains metrics and, when requested, a classification report.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ==================================
# Create DataLoader
# ==================================
def create_dataloader(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
) -> DataLoader:
    """
    Create a DataLoader for a tokenized evaluation dataset.
    """
    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, (tuple, list)) and model_output:
        return model_output[0]
    raise TypeError("Model output does not contain logits.")


def _unpack_batch(
    batch: Mapping[str, torch.Tensor] | tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        return (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )

    input_ids, attention_mask, labels = batch
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def _build_classification_report(
    all_labels: list[int],
    all_preds: list[int],
    label_mapping: Mapping[str, int] | None = None,
) -> str:
    if not all_labels:
        return "No labels available for classification report."

    if label_mapping:
        sorted_labels = sorted(label_mapping.items(), key=lambda item: item[1])
        report_label_ids = [label_id for _, label_id in sorted_labels]
        target_names = [label for label, _ in sorted_labels]
        return classification_report(
            all_labels,
            all_preds,
            labels=report_label_ids,
            target_names=target_names,
            zero_division=0,
        )

    return classification_report(all_labels, all_preds, zero_division=0)


def calculate_metrics(
    all_labels: list[int],
    all_preds: list[int],
    label_mapping: Mapping[str, int] | None = None,
    include_classification_report: bool = False,
) -> dict[str, Any]:
    """
    Calculate classification metrics from true labels and predictions.
    """
    if not all_labels:
        results: dict[str, Any] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }
        if include_classification_report:
            results["classification_report"] = _build_classification_report(
                all_labels,
                all_preds,
                label_mapping,
            )
        return results

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0,
        ),
    }
    if include_classification_report:
        results["classification_report"] = _build_classification_report(
            all_labels,
            all_preds,
            label_mapping,
        )
    return results


def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
    label_mapping: Mapping[str, int] | None = None,
    include_classification_report: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a model using an existing DataLoader.
    """
    evaluation_device = _resolve_device(device)
    model.to(evaluation_device)

    was_training = model.training
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = _unpack_batch(batch, evaluation_device)

            logits = _extract_logits(
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            )
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    if was_training:
        model.train()

    return calculate_metrics(
        all_labels,
        all_preds,
        label_mapping=label_mapping,
        include_classification_report=include_classification_report,
    )


# ==================================
# Evaluation Function
# ==================================
def evaluate_model(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    label_mapping: Mapping[str, int] | None = None,
    include_classification_report: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a BERT model on tokenized tensors.
    """
    dataloader = create_dataloader(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        batch_size=batch_size,
    )
    results = evaluate_dataloader(
        model,
        dataloader,
        device=device,
        label_mapping=label_mapping,
        include_classification_report=include_classification_report,
    )

    print(
        "Evaluation Results: "
        f"Accuracy = {results['accuracy']:.4f}, "
        f"Precision = {results['precision']:.4f}, "
        f"Recall = {results['recall']:.4f}, "
        f"F1 Score = {results['f1_score']:.4f}"
    )
    if include_classification_report:
        print("Classification Report:")
        print(results["classification_report"])
    return results


if __name__ == "__main__":
    print("Running evaluation...")

    from transformers import AutoTokenizer

    from src.bert.data_prep import prepare_bert_splits
    from src.bert.model import load_checkpoint, load_model
    from src.bert.tokenizer import load_dataset

    df = load_dataset()
    splits = prepare_bert_splits(df)
    texts = splits.test_texts or splits.val_texts or splits.train_texts
    labels = splits.test_labels or splits.val_labels or splits.train_labels

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )

    label_tensor = torch.tensor(labels, dtype=torch.long)
    num_labels = len(splits.label_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(num_labels=num_labels, device=device)

    checkpoint_path = Path(__file__).resolve().parent.parent.parent / "models" / "bert_model.pt"
    if checkpoint_path.exists():
        try:
            model = load_checkpoint(model, str(checkpoint_path), device=device)
            print(f"Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as exc:
            print(f"Could not load checkpoint at {checkpoint_path}: {exc}")
            print("Evaluating an untrained model with the current label mapping.")
    else:
        print("Checkpoint not found; evaluating an untrained model.")

    final_results = evaluate_model(
        model,
        encoded["input_ids"],
        encoded["attention_mask"],
        label_tensor,
        batch_size=32,
        device=device,
        label_mapping=splits.label_mapping,
    )
    final_metrics = {
        key: value
        for key, value in final_results.items()
        if key != "classification_report"
    }
    print("Final Results:", final_metrics)