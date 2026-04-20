# ==================================
# Evaluation Module
# Mitchell Hughes
# 16/04/2026
# Notes:
# ================================== 
# - This module is responsible for evaluating the performance of the BERT model after training.
# - The main function is `evaluate_model` (line 31), which will run through the entire evaluation process.
# - The output will be the evaluation metrics (accuracy and F1 score) for the model as a dictionary


import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score


#==================================
# Create DataLoader
#==================================
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ==================================
# Evaluation Function
# ==================================
def evaluate_model(model, input_ids, attention_mask, labels, batch_size = 16, device="none"):
    """
    Evaluate the performance of the BERT model on the evaluation dataset.
    Args:
        model (torch.nn.Module): The trained BERT model to evaluate.
        input_ids (torch.Tensor): Tensor of input token IDs for the evaluation dataset.
        attention_mask (torch.Tensor): Tensor of attention masks for the evaluation dataset.
        labels (torch.Tensor): Tensor of labels for the evaluation dataset.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 16.
        device (str, optional): Device to run the evaluation on ("cuda" or "cpu"). Defaults to "none", which will automatically select "cuda" if available.
    Returns:
        dict: A dictionary containing the evaluation metrics (accuracy and F1 score).
    """
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader = create_dataloader(input_ids, attention_mask, labels, batch_size)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]

            logits = model(
                input_ids = batch_input_ids,
                attention_mask = batch_attention_mask
            )

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())

        
        # Calculate Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        results = {
            "accuracy": accuracy,
            "f1_score": f1
        }

        # Print Results
        print(f"Evaluation Results: Accuracy = {accuracy:.4f}, F1 Score = {f1:.4f}")

        return results
