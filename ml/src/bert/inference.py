"""Inference utilities for optimized BERT part condition classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.bert.model import BertClassifier


class PartConditionInference:
	"""Load trained artifacts and run partcondition predictions."""

	def __init__(self, artifacts_dir: str | Path = "models/bert_optimized") -> None:
		self.artifacts_dir = Path(artifacts_dir)
		self.metadata = self._load_metadata()
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		tokenizer_dir = self.artifacts_dir / "tokenizer"
		model_path = self.artifacts_dir / "model.pt"

		if not tokenizer_dir.exists():
			raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
		if not model_path.exists():
			raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
		self.model = BertClassifier(
			model_name=self.metadata["model_name"],
			num_labels=self.metadata["num_labels"],
		)
		state_dict = torch.load(model_path, map_location=self.device)
		self.model.load_state_dict(state_dict)
		self.model.to(self.device)
		self.model.eval()

		self.label_classes: list[str] = self.metadata["label_classes"]
		self.max_length: int = self.metadata["best_hyperparameters"]["max_length"]

	def _load_metadata(self) -> dict[str, Any]:
		metadata_path = self.artifacts_dir / "metadata.json"
		if not metadata_path.exists():
			raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
		return json.loads(metadata_path.read_text(encoding="utf-8"))

	@staticmethod
	def _build_input_text(discrepancy_text: str, part_name: str | None = None) -> str:
		text = (discrepancy_text or "").strip()
		if not text:
			raise ValueError("discrepancy_text cannot be empty.")

		part = (part_name or "").strip()
		if part:
			return f"Part: {part} | Issue: {text}"
		return text

	def predict(
		self,
		discrepancy_text: str,
		part_name: str | None = None,
	) -> dict[str, Any]:
		"""Predict partcondition from discrepancy text and optional part_name."""
		model_input = self._build_input_text(discrepancy_text, part_name)
		encoded = self.tokenizer(
			[model_input],
			padding=True,
			truncation=True,
			max_length=self.max_length,
			return_tensors="pt",
		)

		input_ids = encoded["input_ids"].to(self.device)
		attention_mask = encoded["attention_mask"].to(self.device)

		with torch.no_grad():
			logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
			probabilities = torch.softmax(logits, dim=1).squeeze(0)
			predicted_id = int(torch.argmax(probabilities).item())

		return {
			"input": {
				"discrepancy_text": discrepancy_text,
				"part_name": part_name,
				"model_text": model_input,
			},
			"predicted_partcondition": self.label_classes[predicted_id],
			"predicted_label_id": predicted_id,
			"confidence": float(probabilities[predicted_id].item()),
			"class_probabilities": {
				label: float(probabilities[idx].item())
				for idx, label in enumerate(self.label_classes)
			},
		}

	def predict_batch(self, items: list[dict[str, str | None]]) -> list[dict[str, Any]]:
		"""Batch prediction wrapper over `predict`.

		Each item should include `discrepancy_text` and optional `part_name`.
		"""
		outputs: list[dict[str, Any]] = []
		for item in items:
			outputs.append(
				self.predict(
					discrepancy_text=str(item.get("discrepancy_text") or ""),
					part_name=item.get("part_name"),
				)
			)
		return outputs


def predict_partcondition(
	discrepancy_text: str,
	part_name: str | None = None,
	artifacts_dir: str | Path = "models/bert_optimized",
) -> dict[str, Any]:
	"""Convenience function for single prediction use-cases."""
	pipeline = PartConditionInference(artifacts_dir=artifacts_dir)
	return pipeline.predict(discrepancy_text=discrepancy_text, part_name=part_name)


if __name__ == "__main__":
	demo = PartConditionInference()
	result = demo.predict("no fault found on avionics panel", part_name="avionics control unit")
	print(json.dumps(result, indent=2))

