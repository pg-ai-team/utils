import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class GroupedDataset(Dataset):
    def __init__(self, root_dir: str, groups_config: dict[str, list[str]], transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.group_names = sorted(list(groups_config.keys()))
        self.group_to_idx = {name: idx for idx, name in enumerate(self.group_names)}

        self.disease_to_label = {}
        for group_name, diseases in groups_config.items():
            label_idx = self.group_to_idx[group_name]
            for disease in diseases:
                self.disease_to_label[disease.lower()] = label_idx

        all_files = [
            f for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        ]

        self.samples = []
        for file_name in sorted(all_files):
            file_lower = file_name.lower()

            for disease_name, label_idx in self.disease_to_label.items():
                if disease_name in file_lower:
                    self.samples.append((file_name, label_idx))
                    break  # Dopasowano grupę -> przechodzimy do kolejnego pliku

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, file_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
def get_data_loaders(cfg, transform=None):
    """
    Buduje i zwraca DataLoadery (train, val, test) wykorzystując instancję klasy konfiguracyjnej.
    """
    fixed_set = getattr(cfg, "FIXED_SET", False)

    if fixed_set:
        print("Tryb FIXED: Ładowanie ze struktury folderów train/validation/test.")
        train_dir = os.path.join(cfg.DATA_DIR, 'train')
        val_dir = os.path.join(cfg.DATA_DIR, 'validation')
        test_dir = os.path.join(cfg.DATA_DIR, 'test')

        train_ds = GroupedDataset(train_dir, cfg.GROUPS, transform=transform)
        val_ds   = GroupedDataset(val_dir, cfg.GROUPS, transform=transform)
        test_ds  = GroupedDataset(test_dir, cfg.GROUPS, transform=transform)

    else:
        print("Tryb SEED: Pobieranie z głównego katalogu i podział losowy.")
        full_dataset = GroupedDataset(cfg.DATA_DIR, cfg.GROUPS, transform=transform)

        print(f"Znaleziono {len(full_dataset)} poprawnych zdjęć w {cfg.DATA_DIR}")
        if len(full_dataset) == 0:
            raise ValueError(
                f"Nie znaleziono pasujących obrazów w ścieżce: {cfg.DATA_DIR}. "
                "Sprawdź poprawność ścieżki oraz zawartość dataClasses.json."
            )

        train_split = getattr(cfg, "TRAIN_SPLIT", 0.8)
        val_split   = getattr(cfg, "VAL_SPLIT", 0.1)

        train_size = int(train_split * len(full_dataset))
        val_size   = int(val_split * len(full_dataset))
        test_size  = len(full_dataset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(cfg.SEED)
        )

    batch_size  = getattr(cfg, "BATCH_SIZE", 32)
    num_workers = getattr(cfg, "NUM_WORKERS", 2)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader