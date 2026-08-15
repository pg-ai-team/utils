"""
WandbLogger - shared W&B wrapper for all pg-ai team projects.

Handles:
  - login (reads WANDB_API_KEY from environment)
  - run initialization with project/entity/run-name
  - metric logging helpers
  - model checkpoint artifact upload
"""

import os
from datetime import datetime
from typing import Any

import wandb

from .env import detect_env


def _get_username() -> str:
    return (
        os.environ.get("KAGGLE_USERNAME")
        or os.environ.get("WANDB_USERNAME")
        or os.environ.get("USER")
        or "unknown"
    )


class WandbLogger:
    """
    Thin wrapper around wandb.Run that standardizes how all team
    projects initialize and log to W&B.

    Args:
        project:   W&B project name, e.g. "vit-implatelet"
        entity:    W&B team/org, e.g. "pg-ai-team"
        config:    Config object (has .to_dict()) or plain dict
        tags:      Extra W&B tags
        run_name:  Override auto-generated name
        notes:     Text notes attached to the run

    Example:
        logger = WandbLogger(project="vit-implatelet", entity="pg-ai-team", config=cfg)

        for epoch in range(epochs):
            logger.log({"train/loss": loss, "train/acc": acc}, step=epoch)

        logger.finish()

    Context manager:
        with WandbLogger(...) as logger:
            logger.log({"loss": 0.1})
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        config: Any = None,
        tags: list[str] | None = None,
        run_name: str | None = None,
        notes: str = "",
    ):
        self.project = project
        self.entity = entity
        self._env = detect_env()

        # Resolve config dict
        if config is None:
            cfg_dict = {}
        elif hasattr(config, "to_dict"):
            cfg_dict = config.to_dict()
        elif isinstance(config, dict):
            cfg_dict = config
        else:
            raise TypeError("config must be a dict or have a .to_dict() method")

        # Build run name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self._run_name = run_name or f"{_get_username()}_{timestamp}"

        # Auto-tag with environment
        _tags = list(tags or [])
        _tags.append(self._env)

        # Login
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key, relogin=False)
        else:
            wandb.login()   # falls back to interactive / netrc

        # Init run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self._run_name,
            config=cfg_dict,
            tags=_tags,
            notes=notes,
            resume="allow",
        )

        print(f"[WandbLogger] Run started: {self.run.url}")

    def log(self, metrics: dict[str, Any], step: int | None = None):
        """Log a dict of metrics. Keys should use slash-namespacing: 'train/loss'."""
        wandb.log(metrics, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ):
        payload = {f"train/{k}": v for k, v in train_metrics.items()}
        payload.update({f"val/{k}": v for k, v in val_metrics.items()})
        payload["epoch"] = epoch
        wandb.log(payload, step=epoch)

    def log_test(self, test_metrics: dict[str, float]):
        """Log final test-set metrics (no step — summary only)."""
        payload = {f"test/{k}": v for k, v in test_metrics.items()}
        wandb.log(payload)

        for k, v in payload.items():
            self.run.summary[k] = v

    def log_confusion_matrix(self, y_true, y_pred, class_names: list[str]):
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names,
            )
        })

    def save_model(self, path: str, name: str = "model", metadata: dict | None = None):
        """
        Upload a model checkpoint as a W&B artifact.

        Args:
            path:     Local file path to the checkpoint (.pt / .pth)
            name:     Artifact name (default "model")
            metadata: Optional extra metadata stored on the artifact
        """
        artifact = wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(path)
        self.run.log_artifact(artifact)
        print(f"[WandbLogger] Artifact '{name}' uploaded: {path}")

    def finish(self):
        """Finish the W&B run."""
        if self.run is not None:
            wandb.finish()
            print("[WandbLogger] Run finished.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
