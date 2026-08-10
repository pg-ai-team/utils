"""
Google Colab environment verification script.

Tests:
1. Environment detection ('colab')
2. Secret loading via google.colab.userdata
3. GPU detection (CUDA) & PyTorch 2+2 tensor calculation
4. WandbLogger logging
5. Checkpoint saving to /content/weights
"""

import os
import torch
from pg_ai_utils import detect_env, load_secrets, WandbLogger
from config import ColabConfig


def main():
    print("=" * 60)
    print("VERIFYING GOOGLE COLAB ENVIRONMENT")
    print("=" * 60)

    # 1. Detect environment
    env = detect_env()
    print(f"[1/5] Detected Environment: '{env}'")

    # 2. Load secrets from Colab userdata
    secrets = load_secrets(["WANDB_API_KEY", "KAGGLE_USERNAME"])
    print(f"[2/5] Loaded secrets keys: {list(secrets.keys())}")

    # 3. Instantiate ColabConfig and check GPU / paths
    cfg = ColabConfig()
    cfg.print_summary()

    print(f"[3/5] Hardware Device: {cfg.DEVICE}")
    print(f"      CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU Name: {torch.cuda.get_device_name(0)}")

    # PyTorch 2 + 2 tensor test on resolved device
    tensor_a = torch.tensor([2.0], device=cfg.DEVICE)
    tensor_b = torch.tensor([2.0], device=cfg.DEVICE)
    tensor_res = tensor_a + tensor_b
    print(f"      PyTorch 2+2 Tensor Test on device '{cfg.DEVICE}': {tensor_a.item()} + {tensor_b.item()} = {tensor_res.item()}")
    assert tensor_res.item() == 4.0, "Tensor addition failed!"

    # 4. Initialize W&B logger
    print("[4/5] Testing WandbLogger...")
    with WandbLogger(
        project="verification-colab",
        entity="pg-ai-team",
        config=cfg,
        tags=["verification", "colab"],
        notes="Google Colab verification run",
    ) as logger:
        for epoch in range(cfg.EPOCHS):
            train_metrics = {"loss": 0.55 - epoch * 0.1, "acc": 0.68 + epoch * 0.1}
            val_metrics = {"loss": 0.45 - epoch * 0.1, "acc": 0.72 + epoch * 0.1}
            logger.log_epoch(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)

        # 5. Checkpoint saving test
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
        ckpt_path = os.path.join(cfg.SAVE_DIR, "colab_test_model.pt")
        torch.save({"model_state": "dummy_colab_weights"}, ckpt_path)
        logger.save_model(ckpt_path, name="colab-verification-checkpoint")

    print("=" * 60)
    print("COLAB ENVIRONMENT VERIFICATION SUCCESSFUL!")
    print("=" * 60)


if __name__ == "__main__":
    main()
