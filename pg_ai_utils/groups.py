"""
Class-group definitions: which diseases (file-name prefixes) belong to which class.

A groups mapping is a dict {group_name: [disease, ...]}, e.g.
    {"Neurological": ["Glioma"], "Respiratory": ["Headandneckcancer", "Non-small-celllungcancer"]}

Disease names are the image file-name prefixes (GroupAlternative without spaces).
Bundled presets live in pg_ai_utils/group_presets/*.json and are referenced by name.
"""

import json
import os
from importlib import resources

PRESETS_PACKAGE = "pg_ai_utils"
PRESETS_DIR = "group_presets"


def list_group_presets() -> list[str]:
    """Names of the bundled presets, e.g. ['organ_systems', 'top5_tumor_types']."""
    folder = resources.files(PRESETS_PACKAGE).joinpath(PRESETS_DIR)
    return sorted(p.name[:-5] for p in folder.iterdir() if p.name.endswith(".json"))


def load_groups(spec: str, search_dirs: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """
    Load a groups mapping.

    Args:
        spec:        bundled preset name ("organ_systems") or a path to a JSON file
        search_dirs: extra folders where a relative path is looked up (e.g. DATA_DIR)
    """
    if spec in list_group_presets():
        text = resources.files(PRESETS_PACKAGE).joinpath(PRESETS_DIR, f"{spec}.json").read_text("utf-8")
        groups = json.loads(text)
    else:
        candidates = [spec] + [os.path.join(d, spec) for d in search_dirs if d]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(
                f"Groups '{spec}' is neither a bundled preset {list_group_presets()} "
                f"nor an existing file (checked: {candidates})"
            )
        with open(path, "r", encoding="utf-8") as f:
            groups = json.load(f)
    validate_groups(groups)
    return groups


def validate_groups(groups: dict[str, list[str]]) -> None:
    """A disease may belong to only one group, otherwise its label would be ambiguous."""
    owner: dict[str, str] = {}
    for group, diseases in groups.items():
        for disease in diseases:
            if disease in owner:
                raise ValueError(f"Disease '{disease}' is in both '{owner[disease]}' and '{group}'")
            owner[disease] = group
