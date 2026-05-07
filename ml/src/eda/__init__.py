from pathlib import Path

EDA_DIR = Path(__file__).resolve().parent
EDA_TEMP_DIR = EDA_DIR / "temp"
EDA_PLOTS_DIR = EDA_TEMP_DIR / "plots"

# Keep module-local temp directories available for EDA artifacts.
EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

from .class_imbalance import ClassImbalanceEDA, analyze_class_imbalance

__all__ = [
    "ClassImbalanceEDA",
    "analyze_class_imbalance",
    "EDA_DIR",
    "EDA_TEMP_DIR",
    "EDA_PLOTS_DIR",
]
