"""
Example Kaggle Kernel project configuration.
Inherits environment-aware paths and device resolution from pg_ai_utils.BaseConfig.
"""

from dataclasses import dataclass
from pg_ai_utils import BaseConfig


@dataclass
class KaggleConfig(BaseConfig):
    PROJECT_NAME: str = "kaggle-verification-test"
    DATASET_NAME: str = "skykuba/implatelet"
    DATASET_SUBPATH: str = "Images"
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 3
