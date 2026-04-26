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
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"Model saved to {path}")


def load_checkpoint(
    path="bert_model.pt",
    model_name="bert-base-uncased",
    num_labels=2,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = BertClassifier(
        model_name=model_name,
        num_labels=num_labels
    )

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    print(f"Model loaded from {path}")

    return model

# ==================================
# Optimizer
# ==================================

def get_optimizer(model, lr=5e-5):
    from torch.optim import AdamW
    return AdamW(model.parameters(), lr=lr)

# ==================================
# TEST BLOCK
# ==================================

if __name__ == "__main__":
    print("Testing BERT model...")

    # Dummy input
    batch_size = 2
    seq_length = 10
    num_labels = 3

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    # Load model
    model = load_model(num_labels=num_labels)

    # Forward pass
    outputs = model(input_ids, attention_mask)

    print("Output shape:", outputs.shape)

    # Test optimizer
    optimizer = get_optimizer(model)
    print("Optimizer created:", type(optimizer))

    # Test save/load
    save_checkpoint(model, "test_model.pt")
    model = load_checkpoint("test_model.pt", num_labels=num_labels)

    print("All tests passed ")
