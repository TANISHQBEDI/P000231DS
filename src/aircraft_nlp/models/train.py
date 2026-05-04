from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from aircraft_nlp.models.data_prep import prepare_bert_splits
from aircraft_nlp.models.evaluate import evaluate_dataloader


# =============================
# Dataset
# =============================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# =============================
# Model (DistilBERT)
# =============================
class DistilBertClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)  # mean pooling (better for distilbert)
        x = self.dropout(pooled)
        return self.classifier(x)


# =============================
# Training
# =============================
def train_model(df: pd.DataFrame):

    # ---- splits ----
    splits = prepare_bert_splits(df)
    num_labels = len(splits.label_mapping)

    # ---- config ----
    model_name = "distilbert-base-uncased"
    batch_size = 16
    max_len = 192
    epochs = 10
    lr = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_loader = DataLoader(
        TextDataset(splits.train_texts, splits.train_labels, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TextDataset(splits.val_texts, splits.val_labels, tokenizer, max_len),
        batch_size=batch_size,
    )

    test_loader = DataLoader(
        TextDataset(splits.test_texts, splits.test_labels, tokenizer, max_len),
        batch_size=batch_size,
    )

    # ---- model ----
    model = DistilBertClassifier(model_name, num_labels).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- training loop ----
    best_f1 = -1
    patience = 2
    no_improve = 0
    best_state = None

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            if step % 20 == 0:
                print(f"Step {step}/{len(train_loader)}")

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

        # ---- validation ----
        metrics = evaluate_dataloader(model, val_loader, device)
        f1 = metrics["f1_score"]

        print(f"VAL → Acc: {metrics['accuracy']:.3f} | F1: {f1:.3f}")

        # ---- best model (F1) ----
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # ---- early stopping ----
        if no_improve >= patience:
            print("Early stopping")
            break

    # ---- load best ----
    model.load_state_dict(best_state)

    # ---- final test ----
    print("\nFINAL TEST RESULTS:")

    test_metrics = evaluate_dataloader(
        model,
        test_loader,
        device,
        label_mapping=splits.label_mapping,
        include_classification_report=True,
    )

    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1 Score : {test_metrics['f1_score']:.4f}")

    print("\nClassification Report:")
    print(test_metrics["classification_report"])

    # ---- save everything ----
    Path("models").mkdir(exist_ok=True)

    torch.save(model.state_dict(), "models/model.pt")
    tokenizer.save_pretrained("models/tokenizer")

    with open("models/label_mapping.json", "w") as f:
        json.dump(splits.label_mapping, f)

    print("\nSaved model + tokenizer + labels")

    return model