# ==================================
# Training Module
# Mitchell Hughes
# 16/04/2026
# Notes:
# ================================== 
# - This module is responsible for training the BERT model using the tokenised data.
# - The main function is `train_model` (line 43), which will run through the entire training process.
# - The output will be the trained model, a list of average loss values for each epoch
# - The model will be saved to a file using `save_model`

import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

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

        #  TODO: Update model save path, and function usage below
        # save_model(model)

    return model, loss_history

# ==================================
# Save Model
# ==================================

# TODO: Update save path and frequency
def save_model(model, path = "model.pt"):
    """Save the trained BERT model to a file.
    Args:
        model (torch.nn.Module): The trained BERT model to save.
        path (str, optional): The file path to save the model to. Defaults to "model.pt".
    Returns:
        None
    """

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")