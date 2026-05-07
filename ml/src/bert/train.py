# ==================================
# Training Module
# Mitchell Hughes
# 16/04/2026
# Notes:
# ================================== 
# - This module is responsible for training the BERT model using the tokenised data.
# - The main function is `train_model`, which will run through the entire training process.
# - The output will be the trained model, a list of average loss values for each epoch
# - The model will be saved to a file using `save_model`. It is currently set to save after all epochs are finished.

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

# ==================================
# Configuration
# ==================================

MODEL_SAVE_PATH = Path(__file__).resolve().parent.parent / "models"
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist


# ==================================
# Create DataLoader
# ==================================

def create_dataloader(input_ids, attention_mask, labels, batch_size = 16):
    """
    Create a DataLoader for the evaluation dataset.
    Args:
        input_ids (torch.Tensor): Tensor of input token IDs.
        attention_mask (torch.Tensor): Tensor of attention masks.
        labels (torch.Tensor): Tensor of labels.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 16.
    Returns:
        DataLoader: A DataLoader for the evaluation dataset.
    """

    dataset = TensorDataset(input_ids, attention_mask, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==================================
# Training Function
# ==================================

def train_model(model, input_ids, attention_mask, labels, batch_size=16, epochs=3, lr=5e-5, device=None):
    """
    Train the BERT model on the training dataset.
    Args:
        model (torch.nn.Module): The BERT model to train.
        input_ids (torch.Tensor): Tensor of input token IDs for the training dataset.
        attention_mask (torch.Tensor): Tensor of attention masks for the training dataset.
        labels (torch.Tensor): Tensor of labels for the training dataset.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 16.
        epochs (int, optional): Number of epochs to train for. Defaults to 3.
        lr (float, optional): Learning rate for the optimizer. Defaults to 5e-5.
        device (str, optional): Device to run the training on ("cuda" or "cpu"). Defaults to None, which will automatically select "cuda" if available.
    Returns:
        torch.nn.Module: The trained BERT model.
        list: A list of average loss values for each epoch.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = create_dataloader(input_ids, attention_mask, labels, batch_size)

    optimiser = AdamW(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    loss_history = []

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]

            # Forward Pass
            logits = model(
                input_ids = batch_input_ids,
                attention_mask = batch_attention_mask
            )

            loss = criterion(logits, batch_labels)

            # Backwards Pass
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs}- Loss: {avg_loss:.4f}" )

    #  Uncomment to Save the model after training is complete
    # save_model(model)

    return model, loss_history

# ==================================
# Save Model
# ==================================

def save_model(model, path = MODEL_SAVE_PATH, filename=None):
    """Save the trained BERT model to a file.
    Args:
        model (torch.nn.Module): The trained BERT model to save.
        path (str, optional): The file path to save the model to. Defaults to MODEL_SAVE_PATH which is defined at the top of this script.
        filename (str, optional): The name of the file to save the model to. If not provided, a timestamp will be used.
    Returns:
        None
    """
    # Save the model state using timestamp filename unless a name is provided
    if filename is None:
        filename = f"_{torch.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

    torch.save(model.state_dict(), path + filename)
    print(f"Model saved to {path + filename}")

def train_model(model, input_ids, attention_mask, labels, batch_size=16, epochs=3, lr=5e-5, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = create_dataloader(input_ids, attention_mask, labels, batch_size)

    optimiser = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []

    # START PRINT
    print("\n===== TRAINING STARTED =====")
    print(f"Total samples: {len(input_ids)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print("===========================\n")

    for epoch in range(epochs):
        total_loss = 0

        # progress part
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]

            logits = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )

            loss = criterion(logits, batch_labels)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"[Epoch {epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")

    # END PRINT
    print("\n===== TRAINING COMPLETE =====")
    print("Loss history:", loss_history)
    print("============================\n")

    return model, loss_history

if __name__ == "__main__":
    print("Loading data...")

    from ml.src.bert.tokenizer import load_bert_tokenizer_inputs
    from ml.src.bert.model import load_model
    from transformers import AutoTokenizer

    # Load raw text + labels
    texts, labels, metadata = load_bert_tokenizer_inputs()

    # Limit data (important — your dataset is huge)
    texts = texts[:1000]
    labels = labels[:1000]

    print(f"Loaded {len(texts)} samples")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    import torch
    labels = torch.tensor(labels, dtype=torch.long)

    print("Initializing model...")

    num_labels = max(labels.tolist()) + 1
    model = load_model(num_labels=num_labels)

    print("Training started...")

    model, history = train_model(
        model,
        input_ids,
        attention_mask,
        labels,
        batch_size=32,
        epochs=1
    )

    print("Training complete")
    print("Loss history:", history)
