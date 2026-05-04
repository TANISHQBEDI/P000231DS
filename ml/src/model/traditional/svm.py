import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.model.base import BaseModel
from src.utils.paths import PROCESSED_DIR
from sklearn.svm import LinearSVC

class SVMModel(BaseModel):
    """SVM classifier for text classification."""

    def __init__(self):
        self.model = LinearSVC()
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def train(self, X, y, **kwargs):
        """Train SVM model."""
        if isinstance(X, list):
            X = self.vectorizer.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions."""
        if isinstance(X, list):
            X = self.vectorizer.transform(X)
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate model."""
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        }

    def save(self, path=None):
        """Save model to standard location."""
        model_dir = PROCESSED_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        if path is None:
            path = model_dir / "svm_model.pkl"

        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "vectorizer": self.vectorizer
            }, f)

        print(f"SVM model saved to: {path}")

    def load(self, path):
        """Load model from standard location."""
        model_dir = PROCESSED_DIR / "models"

        if path is None:
            path = model_dir / "svm_model.pkl"

        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.vectorizer = data["vectorizer"]

        print(f"SVM model loaded from: {path}")
        return self
