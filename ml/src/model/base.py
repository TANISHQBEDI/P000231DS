from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate the model."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
