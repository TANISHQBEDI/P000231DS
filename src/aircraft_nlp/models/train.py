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

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

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
        
        # Freeze all embeddings and transformer layers first
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Unfreeze ONLY the last 2 transformer blocks (layers 4 and 5)
        for param in self.bert.transformer.layer[4:].parameters():
            param.requires_grad = True

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

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    train_labels_array = np.array(splits.train_labels)
    classes = np.unique(train_labels_array)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels_array
    )

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # ---- training loop ----
    best_f1 = -1
    patience = 2
    no_improve = 0
    best_state = None

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    

    history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "train_precision": [],
    "val_precision": [],
    "train_recall": [],
    "val_recall": [],
    "train_f1": [],
    "val_f1": [],
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- validation loss ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids, mask)
                loss = loss_fn(logits, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # ---- metrics ----
        train_metrics = evaluate_dataloader(model, train_loader, device)
        val_metrics = evaluate_dataloader(model, val_loader, device)

        # ---- store history ----
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])

        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])

        history["train_f1"].append(train_metrics["f1_score"])
        history["val_f1"].append(val_metrics["f1_score"])

        # ---- print every epoch ----
        print(f"\nEpoch {epoch+1}")
        print("-" * 58)
        print(f"{'Split':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Loss':<8}")
        print("-" * 58)
        print(f"{'Train':<8} {train_metrics['accuracy']:<8.4f} {train_metrics['precision']:<8.4f} {train_metrics['recall']:<8.4f} {train_metrics['f1_score']:<8.4f} {avg_train_loss:<8.4f}")
        print(f"{'Test':<8} {val_metrics['accuracy']:<8.4f} {val_metrics['precision']:<8.4f} {val_metrics['recall']:<8.4f} {val_metrics['f1_score']:<8.4f} {avg_val_loss:<8.4f}")
        print("-" * 58)

        # ---- optional special print every 20 epochs ----
        if (epoch + 1) % 20 == 0:
            print("\n--- 20 epoch checkpoint reached ---\n")

        # ---- best model (based on val F1) ----
        f1 = val_metrics["f1_score"]
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