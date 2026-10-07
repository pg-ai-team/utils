"""
Grouped image dataset and DataLoader builder for KEGG pathway images.

File-name convention: <Disease>_<SampleID>.png, e.g. Breastcancer_MGH-BrCa-86-TR1197.png
Each disease is mapped to a class group by cfg.GROUPS (see pg_ai_utils.groups).

Split controls (all optional, fields of BaseConfig):
    EXCLUDE_GROUPS       groups removed from the dataset entirely
    EXCLUDE_DISEASES     diseases removed from the dataset entirely
    TRAIN_ONLY_GROUPS    groups used for training but never in validation / test
    HOLDOUT_DISEASES     diseases placed only in the test set (leave-one-group-out);
                         their samples keep the label of their group
    VAL_SPLIT/TEST_SPLIT fractions, applied per group (stratified), seeded with SEED
"""

import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class Sample:
    file_name: str
    sample_id: str
    disease: str
    group: str


def red_channel(image: Image.Image) -> Image.Image:
    """Single-channel image with the R channel (the only informative one in KEGG pathway PNGs).

    A module-level function (not a lambda), so transforms stay picklable for DataLoader workers
    started with 'spawn' / 'forkserver' (macOS, Windows, Linux with Python >= 3.14).
    """
    return image.getchannel("R")


def parse_file_name(file_name: str) -> tuple[str, str]:
    """'Breastcancer_MGH-BrCa-86-TR1197.png' -> ('Breastcancer', 'MGH-BrCa-86-TR1197')"""
    stem = os.path.splitext(file_name)[0]
    if "_" not in stem:
        raise ValueError(f"File name '{file_name}' does not follow <Disease>_<SampleID>")
    disease, sample_id = stem.split("_", 1)
    return disease, sample_id


def scan_samples(
    root_dir: str,
    groups: dict[str, list[str]],
    exclude_groups: tuple[str, ...] = (),
    exclude_diseases: tuple[str, ...] = (),
) -> list[Sample]:
    """List image files and assign each one to its group by EXACT disease name."""
    disease_to_group = {d: g for g, diseases in groups.items() for d in diseases}
    samples, unmapped = [], Counter()

    for file_name in sorted(os.listdir(root_dir)):
        if not file_name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        disease, sample_id = parse_file_name(file_name)
        group = disease_to_group.get(disease)
        if group is None:
            unmapped[disease] += 1
            continue
        if group in exclude_groups or disease in exclude_diseases:
            continue
        samples.append(Sample(file_name, sample_id, disease, group))

    if unmapped:
        print(f"[dataset] Skipped {sum(unmapped.values())} files with diseases not in GROUPS: {dict(unmapped)}")
    return samples


def stratified_split(
    samples: list[Sample],
    val_split: float,
    test_split: float,
    seed: int,
    train_only_groups: tuple[str, ...] = (),
    holdout_diseases: tuple[str, ...] = (),
) -> dict[str, list[Sample]]:
    """Seeded split applied separately within each group, so class proportions are preserved."""
    rng = random.Random(seed)
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    by_group: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        if s.disease in holdout_diseases:
            splits["test"].append(s)
        else:
            by_group[s.group].append(s)

    for group in sorted(by_group):
        members = by_group[group]
        rng.shuffle(members)
        if group in train_only_groups:
            splits["train"].extend(members)
            continue
        n_test = round(len(members) * test_split)
        n_val = round(len(members) * val_split)
        splits["test"].extend(members[:n_test])
        splits["val"].extend(members[n_test : n_test + n_val])
        splits["train"].extend(members[n_test + n_val :])
    return splits


class GroupedDataset(Dataset):
    """Images of one split. Labels are indices into class_names (sorted group names)."""

    def __init__(
        self,
        root_dir: str,
        samples: list[Sample],
        class_names: list[str],
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.samples = samples
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Returns (image, label); image is a PIL image unless transform converts it (e.g. ToTensor)."""
        sample = self.samples[idx]
        with Image.open(os.path.join(self.root_dir, sample.file_name)) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[sample.group]


def get_data_loaders(
    cfg,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from cfg.DATA_DIR and cfg.GROUPS.

    train_transform is applied to the training set only (put augmentation here);
    eval_transform to validation and test. Class names: loader.dataset.class_names.
    """
    if not cfg.GROUPS:
        raise ValueError("cfg.GROUPS is empty — set GROUPS_FILE (preset name or path) or GROUPS")

    samples = scan_samples(
        cfg.DATA_DIR,
        cfg.GROUPS,
        exclude_groups=tuple(cfg.EXCLUDE_GROUPS),
        exclude_diseases=tuple(cfg.EXCLUDE_DISEASES),
    )
    if not samples:
        raise ValueError(f"No images matching GROUPS found in {cfg.DATA_DIR}")

    splits = stratified_split(
        samples,
        cfg.VAL_SPLIT,
        cfg.TEST_SPLIT,
        cfg.SEED,
        train_only_groups=tuple(cfg.TRAIN_ONLY_GROUPS),
        holdout_diseases=tuple(cfg.HOLDOUT_DISEASES),
    )
    class_names = sorted({s.group for s in samples})

    for name, part in splits.items():
        print(f"[dataset] {name}: {len(part)} samples {dict(sorted(Counter(s.group for s in part).items()))}")

    pin_memory = cfg.DEVICE.type == "cuda"
    batch_size = getattr(cfg, "BATCH_SIZE", 32)
    num_workers = getattr(cfg, "NUM_WORKERS", 2)

    def loader(split: str, transform: Optional[Callable], shuffle: bool) -> DataLoader:
        dataset = GroupedDataset(cfg.DATA_DIR, splits[split], class_names, transform)
        generator = torch.Generator().manual_seed(cfg.SEED) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )

    return (
        loader("train", train_transform, shuffle=True),
        loader("val", eval_transform, shuffle=False),
        loader("test", eval_transform, shuffle=False),
    )
