"""
BaseConfig — environment-aware base configuration class.

Each project subclasses this and fills in dataset-specific fields.
"""

import json
import os
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .env import detect_env


@dataclass
class BaseConfig:
    """
    Base configuration with environment-aware paths and device setup.

    Usage:
        class Config(BaseConfig):
            DATASET_NAME: str = "skykuba/implatelet"
            NUM_CLASSES: int = 2
            ...

        cfg = Config()
        print(cfg.DATA_DIR, cfg.DEVICE)
    """

    # ── Dataset (must be set by subclass) ──────────────────────────────
    DATASET_NAME: str = ""          # e.g. "skykuba/implatelet"
    DATASET_SUBPATH: str = ""       # subfolder inside the downloaded dataset
    GROUPS_FILE: str = "../dataClasses.json"
    GROUPS: Dict[str, List[str]] = field(init=False)
    NUM_CLASSES: int = field(init=False)

    # ── Paths (auto-resolved per environment) ──────────────────────────
    DATA_DIR: str = field(init=False)
    SAVE_DIR: str = field(init=False)
    ENV: str = field(init=False)

    # ── Device ─────────────────────────────────────────────────────────
    DEVICE: torch.device = field(init=False)

    # ── Reproducibility ────────────────────────────────────────────────
    SEED: int = 42

    def __post_init__(self):
        self.ENV = detect_env()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DATA_DIR = self._resolve_data_dir()
        self.SAVE_DIR = self._resolve_save_dir()

        self.GROUPS = self._load_groups()
        self.NUM_CLASSES = len(self.GROUPS)

    def _load_groups(self) -> Dict[str, List[str]]:
            """Ładuje mapowanie grup z pliku JSON (szuka w DATA_DIR lub ścieżce roboczej)."""
            possible_paths = [
                os.path.join(self.DATA_DIR, self.GROUPS_FILE),
                self.GROUPS_FILE,
            ]
            target_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    target_path = path
                    break
            if not target_path:
                raise FileNotFoundError(
                    f"Nie znaleziono pliku klas/grup '{self.GROUPS_FILE}'. "
                    f"Sprawdzono ścieżki: {possible_paths}"
                )

            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _resolve_data_dir(self) -> str:
        if self.ENV == "kaggle":
            # Kaggle: data is at /kaggle/input/<dataset-slug>/
            slug = self.DATASET_NAME.replace("/", "-") if self.DATASET_NAME else "dataset"
            base = f"/kaggle/input/{slug}"
            return os.path.join(base, self.DATASET_SUBPATH) if self.DATASET_SUBPATH else base

        elif self.ENV == "colab":
            if not self.DATASET_NAME:
                raise ValueError("DATASET_NAME must be set for Colab (needed for kagglehub)")
            import kagglehub
            path = kagglehub.dataset_download(self.DATASET_NAME)
            return os.path.join(path, self.DATASET_SUBPATH) if self.DATASET_SUBPATH else path

        else:  # local
            base = os.environ.get("DATA_DIR", self.DATASET_SUBPATH or "data")
            return base

    def _resolve_save_dir(self) -> str:
        dirs = {
            "kaggle": "/kaggle/working/weights",
            "colab": "/content/weights",
            "local": "weights_local",
        }
        return dirs.get(self.ENV, "weights")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to flat dict (for wandb.init config=)."""
        result = {}
        for key in self.__dataclass_fields__:
            # Sprawdzamy, czy atrybut w ogóle istnieje w instancji
            if hasattr(self, key):
                val = getattr(self, key)
                if isinstance(val, torch.device):
                    val = str(val)
                result[key.lower()] = val
        return result

    def print_summary(self):
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        for key, val in self.to_dict().items():
            print(f"  {key}: {val}")
        print("=" * 50)
