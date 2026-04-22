# ==================================
# BERT Model Module
# ==================================

import torch
import torch.nn as nn
from transformers import AutoModel


class BertClassifier(nn.Module):
    """
    BERT-based text classification model
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.3
    ):
        """
        Parametrised constructor

        Parameters:
        model_name (str): pretrained model name
        num_labels (int): number of output classes
        dropout (float): dropout rate
        """

        super(BertClassifier, self).__init__()

        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)

        # Hidden size from BERT
        hidden_size = self.bert.config.hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Classification layer
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass

        Parameters:
        input_ids: token ids
        attention_mask: mask

        Returns:
        logits
        """

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token output
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout
        x = self.dropout(cls_output)

        # Classification
        logits = self.classifier(x)

        return logits


# ==================================
# Utility functions
# ==================================

def load_model(
    model_name="bert-base-uncased",
    num_labels=2,
    device="cpu"
):
    """
    Load model and move to device
    """

    model = BertClassifier(
        model_name=model_name,
        num_labels=num_labels
    )

    model.to(device)

    return model


def save_checkpoint(model, path="bert_model.pt"):
    """
    Save model weights
    """
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path="bert_model.pt", device="cpu"):
    """
    Load model weights
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
