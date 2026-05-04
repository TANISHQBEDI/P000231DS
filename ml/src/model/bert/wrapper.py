import torch
from transformers import AutoTokenizer
from src.model.base import BaseModel
from src.model.bert.bert import BertClassifier, save_checkpoint
from src.utils.paths import PROCESSED_DIR
from src.model.bert.bert import load_checkpoint

class BertModel(BaseModel):
    """BERT model wrapper implementing BaseModel interface."""
    
    def __init__(self, model_name="bert-base-uncased", num_labels=2, device=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BertClassifier(model_name=model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def train(self, texts, labels, batch_size=16, epochs=3, lr=5e-5, **kwargs):
        """Train BERT model."""
        from src.model.training.bert_trainer import train_model
        
        # Tokenize
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # Train
        self.model, loss_history = train_model(
            self.model, input_ids, attention_mask, labels_tensor,
            batch_size=batch_size, epochs=epochs, lr=lr, device=self.device
        )
        return loss_history
    
    def predict(self, texts):
        """Make predictions."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        return predictions
    
    def evaluate(self, texts, labels):
        """Evaluate model."""
        from src.model.evaluation.evaluate import evaluate_model
        
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return evaluate_model(
            self.model, input_ids, attention_mask, labels_tensor,
            device=self.device
        )
    
    def save(self, path: str=None):
        """Save model to standard location."""
        model_dir = PROCESSED_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if path is None:
            path = model_dir / "bert_model.pt"
        
        save_checkpoint(self.model, path)
        print(f"BERT model saved to {path}")
    
    def load(self, path: str):
        """Load model."""
        from src.model.bert.bert import load_checkpoint
        model_dir = PROCESSED_DIR / "models"
        
        if path is None:
            path = model_dir / "bert_model.pt"
            
            self.model = load_checkpoint(path, self.model_name, self.num_labels, self.device)
            print(f"BERT model loaded from: {path}")
            
            return self

