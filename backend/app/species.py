"""Derive species class from folder layout: root/<SpeciesName>/image.jpg"""
from __future__ import annotations

import re
from pathlib import Path

# Generic buckets — not a species label for folder_primary / open-vocab hints
GENERIC_FOLDER_NAMES = frozenset({
    "uncategorized",
    "unclassified",
    "misc",
    "miscellaneous",
    "general",
    "other",
    "others",
    "default",
    "new folder",
    "temp",
    "tmp",
})

SKIP_FOLDER_NAMES = frozenset({
    "images",
    "image",
    "img",
    "imgs",
    "pictures",
    "photos",
    "labels",
    "label",
    "annotations",
    "train",
    "val",
    "valid",
    "validation",
    "test",
    "testing",
    "data",
    "dataset",
    "raw",
    "export",
    "__pycache__",
    "assets",
    "downloads",
    "desktop",
    "documents",
})


def is_valid_species_folder(name: str | None, project_name: str | None = None) -> bool:
    """
    True when parent folder name is a real species folder, not project title / generic path.
    """
    if not name or not str(name).strip():
        return False
    n = str(name).strip()
    low = n.lower()
    if low == "unknown":
        return False
    if low in GENERIC_FOLDER_NAMES:
        return False
    if low in SKIP_FOLDER_NAMES:
        return False
    if project_name and low == project_name.strip().lower():
        return False
    # file-like names (Untitled design.png parent tricks)
    if "." in n and len(n) < 48:
        return False
    return True


def species_from_filename(stem: str) -> str | None:
    """
    YOLO export layout: images/train/gymnothorax_undulatus_image_boost_63.jpg
    -> gymnothorax_undulatus
    """
    s = stem.strip().lower()
    if not s:
        return None
    m = re.match(r"^([a-z][a-z0-9]*_[a-z][a-z0-9]*)", s)
    if not m:
        return None
    candidate = m.group(1)
    if candidate in SKIP_FOLDER_NAMES or candidate in GENERIC_FOLDER_NAMES:
        return None
    return candidate


def species_from_image_path(image_path: str, project_root: str | None = None) -> str:
    """
    Class name = immediate parent folder of the image file.
    e.g. D:/dataset/Clownfish/photo.jpg -> Clownfish
  When parent is train/val/test, infer genus_species from filename.
    """
    p = Path(image_path)
    proj_name = Path(project_root).name if project_root else None
    parent_name = p.parent.name
    if parent_name and is_valid_species_folder(parent_name, proj_name):
        return parent_name.strip()

    if project_root:
        try:
            rel = Path(image_path).relative_to(project_root)
            parts = [
                x
                for x in rel.parts[:-1]
                if is_valid_species_folder(x, proj_name)
            ]
            if parts:
                return parts[-1].strip()
        except ValueError:
            pass

    from_name = species_from_filename(p.stem)
    if from_name and is_valid_species_folder(from_name, proj_name):
        return from_name

    return "unknown"


def discover_species_classes(files: list[dict], project_root: str) -> list[str]:
    names: set[str] = set()
    root = Path(project_root)
    for f in files:
        path = f.get("path", "")
        if not path:
            continue
        folder = f.get("folder")
        if folder:
            name = Path(folder).name
            if name.lower() not in SKIP_FOLDER_NAMES:
                names.add(name.strip())
                continue
        sp = species_from_image_path(path, project_root)
        if sp != "unknown":
            names.add(sp)
    return sorted(names)
