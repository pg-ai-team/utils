"""
Local environment verification script.

Tests:
1. Environment detection ('local')
2. Secret loading from .env
3. Hardware device resolution & PyTorch 2+2 tensor calculation
4. WandbLogger logging
5. Checkpoint saving to local weights folder
"""

import os
import torch
import torchvision.transforms as transforms
from pg_ai_utils import detect_env, load_secrets, WandbLogger, get_data_loaders, red_channel
from config import LocalConfig


def main():
    print("=" * 60)
    print("VERIFYING LOCAL ENVIRONMENT")
    print("=" * 60)

    # 1. Detect environment
    env = detect_env()
    print(f"[1/5] Detected Environment: '{env}'")

    # 2. Load secrets from .env
    secrets = load_secrets(["WANDB_API_KEY", "KAGGLE_USERNAME"])
    print(f"[2/5] Loaded secret keys from environment: {list(secrets.keys())}")

    # 3. Instantiate LocalConfig and check hardware & paths
    cfg = LocalConfig()
    cfg.print_summary()

    # Optional: grouped DataLoaders (only when the dataset is available locally)
    if cfg.GROUPS and os.path.isdir(cfg.DATA_DIR):
        eval_transform = transforms.Compose([red_channel, transforms.ToTensor()])
        train_loader, val_loader, test_loader = get_data_loaders(
            cfg, train_transform=eval_transform, eval_transform=eval_transform
        )
        images, labels = next(iter(train_loader))
        print(
            f"[data] DataLoaders OK: batch {tuple(images.shape)}, "
            f"classes {train_loader.dataset.class_names}"
        )
    else:
        print(f"[data] Skipping DataLoader test (no GROUPS or missing DATA_DIR '{cfg.DATA_DIR}')")

    # PyTorch 2 + 2 tensor test on resolved device (CUDA / MPS / CPU)
    tensor_a = torch.tensor([2.0], device=cfg.DEVICE)
    tensor_b = torch.tensor([2.0], device=cfg.DEVICE)
    tensor_res = tensor_a + tensor_b
    print(f"[3/5] PyTorch 2+2 Tensor Test on device '{cfg.DEVICE}': {tensor_a.item()} + {tensor_b.item()} = {tensor_res.item()}")
    assert tensor_res.item() == 4.0, "Tensor addition failed!"

    # 4. Initialize W&B logger (wandb run)
    print("[4/5] Testing WandbLogger...")
    with WandbLogger(
        project="verification-local",
        entity=None,
        config=cfg,
        tags=["verification", "local"],
        notes="Local environment verification run",
    ) as logger:
        for epoch in range(cfg.EPOCHS):
            train_metrics = {"loss": 0.5 - epoch * 0.1, "acc": 0.7 + epoch * 0.1}
            val_metrics = {"loss": 0.4 - epoch * 0.1, "acc": 0.75 + epoch * 0.1}
            logger.log_epoch(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)

        # 5. Checkpoint saving test
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
        ckpt_path = os.path.join(cfg.SAVE_DIR, "local_test_model.pt")
        torch.save({"model_state": "dummy_local_weights"}, ckpt_path)
        logger.save_model(ckpt_path, name="local-verification-checkpoint")

    print("=" * 60)
    print("LOCAL ENVIRONMENT VERIFICATION SUCCESSFUL!")
    print("=" * 60)


if __name__ == "__main__":
    main()
