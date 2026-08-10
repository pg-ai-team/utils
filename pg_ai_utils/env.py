"""
Environment detection and secret loading.
Supports: Kaggle, Google Colab, Local (.env)
"""

import os
import sys


def detect_env() -> str:
    """Detect current training environment."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "google.colab" in sys.modules:
        return "colab"
    return "local"


def load_secrets(keys: list[str], env: str | None = None) -> dict[str, str]:
    """
    Load secrets from environment-specific secret stores.

    Args:
        keys: List of secret names to load (e.g. ["WANDB_API_KEY", "KAGGLE_USERNAME"])
        env: Override environment detection. One of "kaggle", "colab", "local".

    Returns:
        Dict of {key: value} for found secrets.
    """
    if env is None:
        env = detect_env()

    loaded = {}

    if env == "kaggle":
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            for key in keys:
                try:
                    val = client.get_secret(key)
                    os.environ[key] = val
                    loaded[key] = val
                except Exception:
                    print(f"[pg-ai-utils] Warning: Kaggle secret '{key}' not found")
        except ImportError:
            print("[pg-ai-utils] Warning: kaggle_secrets not available")

    elif env == "colab":
        try:
            from google.colab import userdata
            for key in keys:
                try:
                    val = userdata.get(key)
                    os.environ[key] = val
                    loaded[key] = val
                except Exception:
                    print(f"[pg-ai-utils] Warning: Colab secret '{key}' not found")
        except ImportError:
            print("[pg-ai-utils] Warning: google.colab not available")

    else:  # local
        try:
            from dotenv import load_dotenv
            load_dotenv()
            for key in keys:
                val = os.environ.get(key)
                if val:
                    loaded[key] = val
                else:
                    print(f"[pg-ai-utils] Warning: env var '{key}' not set in .env")
        except ImportError:
            for key in keys:
                val = os.environ.get(key)
                if val:
                    loaded[key] = val

    return loaded
