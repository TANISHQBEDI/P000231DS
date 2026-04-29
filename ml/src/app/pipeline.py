import logging

from src.pipeline.training_pipeline import TrainingConfig, train
from src.utils.paths import PROJECT_ROOT, RAW_FILE

logger = logging.getLogger(__name__)


def run_training(file_path: str = str(RAW_FILE)) -> dict[str, str]:
    mapping_path = PROJECT_ROOT / "src" / "config" / "label_mapping.json"
    config = TrainingConfig(
        data_path=str(file_path),
        mapping_path=str(mapping_path),
    )
    return train(config)


if __name__ == "__main__":
    run_training()
