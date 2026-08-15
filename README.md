# pg-ai-utils

Shared machine learning training utilities for the **pg-ai team**.  
Provides seamless environment detection (**Kaggle**, **Google Colab**, **Local**), standardized configuration management, secret loading, and unified Weights & Biases (`wandb`) experiment logging.

---

## Key Features

- **Automatic Environment Detection (`env.py`)**: Seamlessly identifies whether execution is occurring in **Kaggle Kernels**, **Google Colab**, or a **Local** workstation.
- **Unified Secret Management (`load_secrets`)**: Safely fetches secrets (`WANDB_API_KEY`, `KAGGLE_USERNAME`, etc.) from Kaggle Secrets, Colab `userdata`, or local `.env` files.
- **Environment-Aware Base Configuration (`BaseConfig`)**: Base dataclass exported by `pg-ai-utils` that automatically resolves dataset paths and checkpoint storage per environment and configures PyTorch device (`cuda` / `cpu`).
- **Standardized W&B Logging (`WandbLogger`)**: Clean context-managed wrapper around `wandb` that standardizes authentication, run naming, epoch/test metric logging, confusion matrix plotting, and checkpoint artifact management.

---

## Architecture & Module Overview

> [!IMPORTANT]
> **Separation of Concerns:** `pg-ai-utils` provides the base class `BaseConfig`. Each project repository using `utils` defines its **own separate `config.py`** by subclassing `BaseConfig`.

| Module | Primary Export | Description |
|---|---|---|
| `env.py` | `detect_env()`, `load_secrets()` | Detects platform (`"kaggle"`, `"colab"`, `"local"`) and populates `os.environ` from platform-native secret stores or local `.env`. |
| `config_base.py` | `BaseConfig` | Extensible `dataclass` providing environment-resolved `DATA_DIR`, `SAVE_DIR`, PyTorch device binding, dictionary serialization, and summary printing. |
| `wandb_logger.py` | `WandbLogger` | Context manager wrapping `wandb.init`, `wandb.log`, epoch/test metrics logging, plot rendering, and artifact uploading. |

---

## Installation

### Installing as a Dependency

To install the latest version directly from GitHub:

```bash
pip install git+https://github.com/pg-ai-team/utils.git
```

To pin a specific released version:

```bash
pip install git+https://github.com/pg-ai-team/utils.git@v0.1.0
```

### Kaggle & Google Colab Setup

Include this command in the initial setup cell of your notebook:

```python
!pip install -q git+https://github.com/pg-ai-team/utils.git@v0.1.0
```

### Local Development Setup

To modify or contribute to `pg-ai-utils` locally:

```bash
git clone https://github.com/pg-ai-team/utils.git
cd utils
pip install -e ".[local]"
```

---

## Local Environment Configuration (`.env`)

When running locally, create a `.env` file in your root project directory (refer to [.env.example](.env.example)):

```env
# Weights & Biases API Key
WANDB_API_KEY=your_wandb_api_key_here

# Kaggle Username (optional for local, used for W&B run name fallback)
KAGGLE_USERNAME=your_kaggle_username_here

# Local Data Directory (optional override for BaseConfig local data path)
DATA_DIR=/path/to/local/data
```

---

## Recommended Downstream Project Structure

Every repository using `pg-ai-utils` (e.g. `resnet-project`) maintains its own project-specific configuration and training scripts:

```
my-model-project/
├── config.py          ← Project-specific Config(BaseConfig) defined in THIS repo
├── train.py           ← Main training script importing your local config.py & pg_ai_utils
├── model.py           ← Model architecture definition
├── dataset.py         ← PyTorch Dataset & Dataloader
├── requirements.txt   ← Depends on pg-ai-utils @ git+https://github.com/pg-ai-team/utils.git@v0.1.0
└── .env               ← Local secrets (git-ignored)
```

---

## Verification Examples

The `examples/` folder contains standalone scripts (not included in the installed package) to verify hardware detection (GPU/MPS/CPU), PyTorch tensor math (`2 + 2`), secret loading, and W&B logging across platforms:

```
utils/
├── pg_ai_utils/       ← Core Python package
├── examples/
│   ├── local/         ← Local environment test (config.py & train.py)
│   ├── kaggle/        ← Kaggle Kernels test (config.py & train.py)
│   └── colab/         ← Google Colab test (config.py & train.py)
```


---

## How to Use It in Your Project

### 1. Create a Project-Specific `config.py`

In **your project repository**, create a `config.py` file and subclass `BaseConfig` from `pg_ai_utils`:

> [!NOTE]
> `BaseConfig` handles platform detection (`ENV`), paths (`DATA_DIR`, `SAVE_DIR`), and hardware device (`DEVICE`). Your subclass defines all project-specific hyperparameters.

```python
# config.py (inside YOUR project repo, NOT in utils)
from dataclasses import dataclass, field
from pg_ai_utils import BaseConfig

@dataclass
class Config(BaseConfig):
    # Dataset metadata (used for Kaggle inputs & kagglehub resolution)
    DATASET_NAME: str = "skykuba/implatelet"
    DATASET_SUBPATH: str = "KEGG_Pathway_Image/Images"

    # Architecture & Hyperparameters
    ARCHITECTURE: str = "ViT-B/16"
    NUM_CLASSES: int = 2
    BATCH_SIZE: int = 32

    # Training Schedules
    PHASE1_EPOCHS: int = 100
    PHASE1_LR: float = 1e-3
    PHASE2_EPOCHS: int = 100
    PHASE2_LR: float = 1e-5
```

### 2. Integrate into Your Project's `train.py`

Import `BaseConfig`-derived `Config` from your local `config.py`, and `load_secrets` / `WandbLogger` from `pg_ai_utils`:

```python
# train.py (inside YOUR project repo)
from pg_ai_utils import load_secrets, WandbLogger
from config import Config  # Local project-specific config

def main():
    # 1. Load secrets from Kaggle, Colab, or local .env
    load_secrets(["WANDB_API_KEY", "KAGGLE_USERNAME"])

    # 2. Build configuration (auto-detects paths & hardware)
    cfg = Config()
    cfg.print_summary()

    # 3. Initialize W&B logger using context manager
    with WandbLogger(
        project="vit-implatelet",
        entity="pg-ai-team",
        config=cfg,
        tags=["phase1", "vit-b16"],
        notes="Progressive fine-tuning run",
    ) as logger:

        # Epoch training loop
        for epoch in range(cfg.PHASE1_EPOCHS):
            # ... perform training & validation epoch ...
            train_metrics = {"loss": 0.35, "acc": 0.88}
            val_metrics = {"loss": 0.42, "acc": 0.84}

            # Log epoch metrics to W&B
            logger.log_epoch(epoch, train_metrics, val_metrics)

        # Upload best model checkpoint as W&B artifact
        logger.save_model(f"{cfg.SAVE_DIR}/best.pt", name="vit-implatelet-best")

        # Log final test evaluation metrics
        test_metrics = {"loss": 0.38, "acc": 0.86, "auc": 0.91}
        logger.log_test(test_metrics)

if __name__ == "__main__":
    main()
```

---

## Projects Using `utils`

`pg-ai-utils` serves as the centralized core dependency across multiple team repositories. Each repository maintains its own custom `Config(BaseConfig)` and training code.

---

## Versioning & Collaboration Workflow

To maintain consistent experiment environments across the team, release tagging is used:

```
pg-ai-team/utils
  ├── main          <- Active development
  ├── v0.1.0 (tag)  <- Stable release used by active projects
  └── v0.2.0 (tag)  <- Next feature release
```

**Workflow for updates:**
1. Create a feature branch and submit a Pull Request to `pg-ai-team/utils`.
2. Following approval and merge into `main`, create and push a git release tag:
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0: Added confusion matrix support"
   git push origin v0.2.0
   ```
3. Update dependent project repositories by bumping the release version tag in `requirements.txt` (`@v0.2.0`).
