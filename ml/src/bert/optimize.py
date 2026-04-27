"""Optimization pipeline for BERT text classification.

This module provides:
- hyperparameter search over learning rate and batch size
- optional class-imbalance handling via weighted cross-entropy
- final model training and artifact export for inference
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.bert.model import BertClassifier


@dataclass
class OptimizationConfig:
	model_name: str = "bert-base-uncased"
	text_column: str = "discrepancy"
	label_column: str = "partcondition"
	context_column: str = "part_name"
	learning_rates: tuple[float, ...] = (2e-5, 3e-5, 5e-5)
	batch_sizes: tuple[int, ...] = (8, 16)
	epochs: int = 2
	max_length: int = 128
	val_size: float = 0.2
	random_state: int = 42
	use_class_weights: bool = True


def _prepare_dataframe(
	df: pd.DataFrame,
	text_column: str,
	label_column: str,
	context_column: str,
) -> pd.DataFrame:
	if df is None or df.empty:
		raise ValueError("Input dataframe is empty.")

	missing = [c for c in (text_column, label_column) if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	prepared = df.copy()
	prepared[text_column] = (
		prepared[text_column]
		.fillna("")
		.astype(str)
		.str.replace(r"\s+", " ", regex=True)
		.str.strip()
	)
	prepared[label_column] = prepared[label_column].fillna("unknown").astype(str).str.strip()

	if context_column in prepared.columns:
		prepared[context_column] = (
			prepared[context_column]
			.fillna("")
			.astype(str)
			.str.replace(r"\s+", " ", regex=True)
			.str.strip()
		)
		prepared["model_text"] = prepared.apply(
			lambda row: (
				f"Part: {row[context_column]} | Issue: {row[text_column]}"
				if row[context_column]
				else row[text_column]
			),
			axis=1,
		)
	else:
		prepared["model_text"] = prepared[text_column]

	prepared = prepared.loc[
		(prepared["model_text"] != "") & (prepared[label_column] != "")
	].copy()

	if prepared.empty:
		raise ValueError("No valid rows remain after preprocessing.")

	labels = sorted(prepared[label_column].unique().tolist())
	label_to_id = {label: idx for idx, label in enumerate(labels)}
	prepared["label_id"] = prepared[label_column].map(label_to_id).astype(int)

	return prepared


def _tokenize_texts(
	tokenizer: AutoTokenizer,
	texts: list[str],
	max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	encoded = tokenizer(
		texts,
		padding=True,
		truncation=True,
		max_length=max_length,
		return_tensors="pt",
	)
	return encoded["input_ids"], encoded["attention_mask"]


def _make_loader(
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
	labels: torch.Tensor,
	batch_size: int,
	shuffle: bool,
) -> DataLoader:
	dataset = TensorDataset(input_ids, attention_mask, labels)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _compute_class_weights(
	labels: list[int],
	num_labels: int,
	device: str,
) -> torch.Tensor:
	counts = np.bincount(np.array(labels), minlength=num_labels)
	counts = np.maximum(counts, 1)
	total = counts.sum()
	weights = total / (num_labels * counts)
	return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_epoch(
	model: BertClassifier,
	dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	device: str,
) -> float:
	model.train()
	total_loss = 0.0

	for batch in dataloader:
		input_ids, attention_mask, labels = [t.to(device) for t in batch]

		logits = model(input_ids=input_ids, attention_mask=attention_mask)
		loss = criterion(logits, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		total_loss += float(loss.item())

	return total_loss / max(len(dataloader), 1)


def _evaluate(
	model: BertClassifier,
	dataloader: DataLoader,
	device: str,
) -> dict[str, float]:
	model.eval()
	all_preds: list[int] = []
	all_labels: list[int] = []

	with torch.no_grad():
		for batch in dataloader:
			input_ids, attention_mask, labels = [t.to(device) for t in batch]
			logits = model(input_ids=input_ids, attention_mask=attention_mask)
			preds = torch.argmax(logits, dim=1)

			all_preds.extend(preds.cpu().tolist())
			all_labels.extend(labels.cpu().tolist())

	if not all_labels:
		return {"macro_f1": 0.0, "weighted_f1": 0.0}

	return {
		"macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
		"weighted_f1": float(f1_score(all_labels, all_preds, average="weighted")),
	}


def optimize_model(
	df: pd.DataFrame,
	config: OptimizationConfig | None = None,
	output_dir: str | Path = "models/bert_optimized",
) -> dict[str, Any]:
	"""Run hyperparameter tuning and save the best model artifacts.

	Returns a dictionary with best config, trial scores, and artifact paths.
	"""

	config = config or OptimizationConfig()
	device = "cuda" if torch.cuda.is_available() else "cpu"

	prepared = _prepare_dataframe(
		df,
		text_column=config.text_column,
		label_column=config.label_column,
		context_column=config.context_column,
	)

	texts = prepared["model_text"].tolist()
	labels = prepared["label_id"].tolist()
	label_classes = sorted(prepared[config.label_column].unique().tolist())
	num_labels = len(label_classes)

	x_train, x_val, y_train, y_val = train_test_split(
		texts,
		labels,
		test_size=config.val_size,
		random_state=config.random_state,
		stratify=labels if len(set(labels)) > 1 else None,
	)

	tokenizer = AutoTokenizer.from_pretrained(config.model_name)
	train_ids, train_mask = _tokenize_texts(tokenizer, x_train, config.max_length)
	val_ids, val_mask = _tokenize_texts(tokenizer, x_val, config.max_length)

	y_train_t = torch.tensor(y_train, dtype=torch.long)
	y_val_t = torch.tensor(y_val, dtype=torch.long)

	trials: list[dict[str, Any]] = []
	best_trial: dict[str, Any] | None = None
	best_state_dict: dict[str, Any] | None = None

	for lr in config.learning_rates:
		for batch_size in config.batch_sizes:
			model = BertClassifier(model_name=config.model_name, num_labels=num_labels).to(device)

			optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
			if config.use_class_weights:
				class_weights = _compute_class_weights(y_train, num_labels, device)
				criterion = nn.CrossEntropyLoss(weight=class_weights)
			else:
				criterion = nn.CrossEntropyLoss()

			train_loader = _make_loader(train_ids, train_mask, y_train_t, batch_size, True)
			val_loader = _make_loader(val_ids, val_mask, y_val_t, batch_size, False)

			last_train_loss = 0.0
			for _ in range(config.epochs):
				last_train_loss = _train_one_epoch(
					model=model,
					dataloader=train_loader,
					optimizer=optimizer,
					criterion=criterion,
					device=device,
				)

			val_metrics = _evaluate(model=model, dataloader=val_loader, device=device)
			trial = {
				"learning_rate": lr,
				"batch_size": batch_size,
				"epochs": config.epochs,
				"train_loss": round(last_train_loss, 6),
				"val_macro_f1": round(val_metrics["macro_f1"], 6),
				"val_weighted_f1": round(val_metrics["weighted_f1"], 6),
			}
			trials.append(trial)

			if best_trial is None or trial["val_macro_f1"] > best_trial["val_macro_f1"]:
				best_trial = trial
				best_state_dict = {
					k: v.detach().cpu().clone()
					for k, v in model.state_dict().items()
				}

	if best_trial is None or best_state_dict is None:
		raise RuntimeError("No optimization trial was completed.")

	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	model_path = out_dir / "model.pt"
	tokenizer_path = out_dir / "tokenizer"
	metadata_path = out_dir / "metadata.json"

	torch.save(best_state_dict, model_path)
	tokenizer.save_pretrained(tokenizer_path)

	metadata = {
		"model_name": config.model_name,
		"num_labels": num_labels,
		"label_classes": label_classes,
		"best_hyperparameters": {
			"learning_rate": best_trial["learning_rate"],
			"batch_size": best_trial["batch_size"],
			"epochs": best_trial["epochs"],
			"max_length": config.max_length,
		},
		"class_imbalance": {
			"enabled": config.use_class_weights,
			"strategy": "weighted_cross_entropy" if config.use_class_weights else "none",
		},
		"text_columns": {
			"text_column": config.text_column,
			"context_column": config.context_column,
			"label_column": config.label_column,
		},
		"trials": trials,
		"training_device": device,
	}

	metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

	return {
		"best_trial": best_trial,
		"trials": trials,
		"artifact_paths": {
			"model": str(model_path),
			"tokenizer": str(tokenizer_path),
			"metadata": str(metadata_path),
		},
		"config": asdict(config),
	}

