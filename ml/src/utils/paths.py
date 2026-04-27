from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_FILE = RAW_DIR / "NLP_Dataset_2026_Expanded.xlsx"


def dataset_slug(file_path: str | Path) -> str:
	"""Build a filesystem-safe dataset name from an input file path."""
	name = Path(file_path).stem.strip().lower()
	slug = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
	return slug or "dataset"


def dataset_report_dir(file_path: str | Path) -> Path:
	"""Return dataset-specific processed EDA directory."""
	return PROCESSED_DIR / "eda" / dataset_slug(file_path)
