"""Service layer: mock wrappers that expose the JSON contract the web app
depends on. Real model/training/storage swap in here without touching the
Flask routes or the React frontend.
"""

from .prediction import run_prediction
from .training import run_training
from .storage import save_predictions, list_history, get_history_item

__all__ = [
    "run_prediction",
    "run_training",
    "save_predictions",
    "list_history",
    "get_history_item",
]
