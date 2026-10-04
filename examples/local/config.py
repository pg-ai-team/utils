"""
Example local project configuration.
Inherits environment-aware paths and device resolution from pg_ai_utils.BaseConfig.
"""

from dataclasses import dataclass
from pg_ai_utils import BaseConfig


@dataclass
class LocalConfig(BaseConfig):
    PROJECT_NAME: str = "local-verification-test"
    DATASET_NAME: str = "skykuba/kegg-pathway-images"
    DATASET_SUBPATH: str = "Images"
    GROUPS_FILE: str = "organ_systems"  # bundled preset, see pg_ai_utils.list_group_presets()
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 2
