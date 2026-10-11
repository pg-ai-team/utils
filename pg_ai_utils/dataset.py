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

Sample filters by metadata (need sampleInfo.csv, joined on the sample id from the file name):
    EXCLUDE_LOCATIONS    RealLocation values removed from the dataset, e.g. ["Institute 5"] (NKI)
    MIN_LIB_SIZE         samples with fewer reads (lib.size) removed, e.g. 100000
    SAMPLE_INFO_FILE     path to sampleInfo.csv ("" = look in DATA_DIR and its parent)

Split source:
    SPLIT_SOURCE         "random" (default): stratified_split with VAL_SPLIT / TEST_SPLIT / SEED;
                         "sample_info": the original split from the sampleInfo.csv
                         columns isTraining / isValidation / isTest (VAL_SPLIT, TEST_SPLIT and SEED are ignored)
"""

import csv
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


SAMPLE_INFO_NAME = "sampleInfo.csv"
SPLIT_COLUMNS = {"train": "isTraining", "val": "isValidation", "test": "isTest"}


def find_sample_info(cfg) -> str:
    """Path of sampleInfo.csv: cfg.SAMPLE_INFO_FILE, else DATA_DIR/sampleInfo.csv or its parent folder."""
    explicit = getattr(cfg, "SAMPLE_INFO_FILE", "")
    candidates = [explicit] if explicit else [
        os.path.join(cfg.DATA_DIR, SAMPLE_INFO_NAME),
        os.path.join(os.path.dirname(os.path.normpath(cfg.DATA_DIR)), SAMPLE_INFO_NAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"EXCLUDE_LOCATIONS / MIN_LIB_SIZE / SPLIT_SOURCE='sample_info' need {SAMPLE_INFO_NAME}; not found at {candidates}. "
        "Set SAMPLE_INFO_FILE (e.g. dataset-generator/data/raw/sampleInfo.csv)."
    )


def load_sample_info(path: str, required: tuple[str, ...] = ("RealLocation", "lib.size")) -> dict[str, dict[str, str]]:
    """{sample id: row} from sampleInfo.csv (sep ';', column 'id' = <SampleID> part of the image file name)."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing = {"id", *required} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} has no column(s) {sorted(missing)}")
        info = {}
        for row in reader:
            if row["id"] in info:
                raise ValueError(f"{path}: duplicated id '{row['id']}'")
            info[row["id"]] = row
    return info


def require_sample_info_rows(samples: list[Sample], sample_info: dict[str, dict[str, str]]) -> None:
    """Every sample must have a row in sampleInfo, so metadata cannot be silently missing for some images."""
    missing = [s.file_name for s in samples if s.sample_id not in sample_info]
    if missing:
        raise ValueError(f"{len(missing)} images have no row in sampleInfo (by id), e.g. {missing[:3]}")


def filter_by_metadata(
    samples: list[Sample],
    sample_info: dict[str, dict[str, str]],
    exclude_locations: tuple[str, ...] = (),
    min_lib_size: int = 0,
) -> list[Sample]:
    """Drop samples from exclude_locations (RealLocation) or with lib.size < min_lib_size.

    Every sample must have a row in sampleInfo (otherwise the filter could not be applied and silently keeping
    the sample would make the dataset depend on file names). Unknown locations raise, so a typo cannot turn the
    filter into a no-op.
    """
    require_sample_info_rows(samples, sample_info)
    known = {row["RealLocation"] for row in sample_info.values()}
    unknown = sorted(set(exclude_locations) - known)
    if unknown:
        raise ValueError(f"EXCLUDE_LOCATIONS {unknown} not in sampleInfo RealLocation values {sorted(known)}")

    kept, by_location, by_lib_size = [], Counter(), Counter()
    for s in samples:
        row = sample_info[s.sample_id]
        if row["RealLocation"] in exclude_locations:
            by_location[s.group] += 1
        elif min_lib_size and float(row["lib.size"].replace(",", ".")) < min_lib_size:
            by_lib_size[s.group] += 1
        else:
            kept.append(s)
    if exclude_locations:
        print(f"[dataset] Excluded {sum(by_location.values())} samples from {list(exclude_locations)}: "
              f"{dict(sorted(by_location.items()))}")
    if min_lib_size:
        print(f"[dataset] Excluded {sum(by_lib_size.values())} samples with lib.size < {min_lib_size}: "
              f"{dict(sorted(by_lib_size.items()))}")
    return kept


def load_samples(cfg, sample_info: Optional[dict[str, dict[str, str]]] = None) -> list[Sample]:
    """scan_samples + metadata filters, all driven by cfg (GROUPS, EXCLUDE_*, MIN_LIB_SIZE)."""
    samples = scan_samples(
        cfg.DATA_DIR,
        cfg.GROUPS,
        exclude_groups=tuple(cfg.EXCLUDE_GROUPS),
        exclude_diseases=tuple(cfg.EXCLUDE_DISEASES),
    )
    exclude_locations = tuple(getattr(cfg, "EXCLUDE_LOCATIONS", ()))
    min_lib_size = getattr(cfg, "MIN_LIB_SIZE", 0)
    if exclude_locations or min_lib_size:
        if sample_info is None:
            sample_info = load_sample_info(find_sample_info(cfg))
        samples = filter_by_metadata(samples, sample_info, exclude_locations, min_lib_size)
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


def sample_info_split(
    samples: list[Sample],
    sample_info: dict[str, dict[str, str]],
    train_only_groups: tuple[str, ...] = (),
    holdout_diseases: tuple[str, ...] = (),
) -> dict[str, list[Sample]]:
    """Original split of In 't Veld et al. 2022 from sampleInfo.csv (isTraining / isValidation / isTest).

    Exactly one column must be 1 for every sample. The split was made for pan-cancer detection (all 2351
    samples), so it is not stratified by the classes of a multiclass setup: check the printed class counts.
    TRAIN_ONLY_GROUPS and HOLDOUT_DISEASES are applied on top (they move samples to train / test).
    """
    require_sample_info_rows(samples, sample_info)
    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        row = sample_info[s.sample_id]
        flags = [split for split, column in SPLIT_COLUMNS.items() if row[column].strip() == "1"]
        if len(flags) != 1:
            raise ValueError(f"Sample '{s.sample_id}' must be in exactly one of {list(SPLIT_COLUMNS.values())}, "
                             f"got {[row[c] for c in SPLIT_COLUMNS.values()]}")
        if s.disease in holdout_diseases:
            splits["test"].append(s)
        elif s.group in train_only_groups:
            splits["train"].append(s)
        else:
            splits[flags[0]].append(s)
    return splits


def split_samples(cfg, samples: list[Sample], sample_info: Optional[dict[str, dict[str, str]]] = None
                  ) -> dict[str, list[Sample]]:
    """train / val / test by cfg.SPLIT_SOURCE ("random" = stratified_split, "sample_info" = sampleInfo.csv)."""
    source = getattr(cfg, "SPLIT_SOURCE", "random")
    if source == "random":
        return stratified_split(
            samples,
            cfg.VAL_SPLIT,
            cfg.TEST_SPLIT,
            cfg.SEED,
            train_only_groups=tuple(cfg.TRAIN_ONLY_GROUPS),
            holdout_diseases=tuple(cfg.HOLDOUT_DISEASES),
        )
    if source == "sample_info":
        if sample_info is None:
            sample_info = load_sample_info(find_sample_info(cfg), required=tuple(SPLIT_COLUMNS.values()))
        return sample_info_split(
            samples,
            sample_info,
            train_only_groups=tuple(cfg.TRAIN_ONLY_GROUPS),
            holdout_diseases=tuple(cfg.HOLDOUT_DISEASES),
        )
    raise ValueError(f"SPLIT_SOURCE must be 'random' or 'sample_info', got '{source}'")


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

    # sampleInfo.csv is read once when a metadata filter or the original split needs it
    needs_info = getattr(cfg, "EXCLUDE_LOCATIONS", ()) or getattr(cfg, "MIN_LIB_SIZE", 0) \
        or getattr(cfg, "SPLIT_SOURCE", "random") == "sample_info"
    sample_info = None
    if needs_info:
        required = ("RealLocation", "lib.size")
        if getattr(cfg, "SPLIT_SOURCE", "random") == "sample_info":
            required += tuple(SPLIT_COLUMNS.values())
        sample_info = load_sample_info(find_sample_info(cfg), required=required)

    samples = load_samples(cfg, sample_info)
    if not samples:
        raise ValueError(f"No images matching GROUPS found in {cfg.DATA_DIR}")

    splits = split_samples(cfg, samples, sample_info)
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
