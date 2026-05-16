"""Load 500+ marine species from species_mapping.csv for YOLO-World and label resolution."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# Generic open-vocab prompts (non-species); listed before species for stable indexing
# For YOLO-World detection in folder datasets — no coral/sponge (they cause boxes on reef)
FISH_DETECTION_PROMPTS = [
    "fish",
    "marine fish",
    "reef fish",
    "tropical fish",
    "pelagic fish",
    "juvenile fish",
    "small fish",
    "ornamental fish",
    "boxfish",
    "cowfish",
    "damselfish",
    "wrasse",
    "goby",
    "cardinalfish",
    "snapper",
    "grouper",
    "tuna",
    "swordfish",
    "shark",
    "ray",
    "jellyfish",
    "octopus",
    "squid",
    "shrimp",
]

GENERIC_MARINE_PROMPTS = FISH_DETECTION_PROMPTS + [
    "crab",
    "lobster",
    "shrimp",
    "prawn",
    "squid",
    "octopus",
    "jellyfish",
    "ray",
    "shark",
    "eel",
    "seahorse",
    "Marine Life",
    "Marine Animals",
    "Marine Creatures",
    "Marine Organisms",
    "Marine Plants",
    "Marine Fungi",
    "Marine Protists",
    "Marine Microorganisms",
    "Marine Microbes",
]


@dataclass(frozen=True)
class SpeciesRecord:
    folder: str
    scientific_name: str
    common_name: str


class SpeciesCatalog:
    def __init__(self) -> None:
        self.records: list[SpeciesRecord] = []
        self.prompts: list[str] = []
        self.prompt_to_folder: dict[str, str] = {}
        self._alias_to_folder: dict[str, str] = {}
        self._load()

    def _register_alias(self, alias: str, folder: str) -> None:
        key = alias.strip().lower()
        if key:
            self._alias_to_folder[key] = folder

    def _load(self) -> None:
        path = settings.species_mapping_csv
        if not path.is_file():
            logger.warning("Species mapping CSV not found: %s", path)
            self.prompts = list(GENERIC_MARINE_PROMPTS)
            return

        seen_folders: set[str] = set()
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folder = (row.get("species_folder") or "").strip()
                sci = (row.get("scientific_name") or "").strip()
                common = (row.get("common_name") or "").strip()
                if not folder:
                    continue
                rec = SpeciesRecord(folder=folder, scientific_name=sci, common_name=common)
                self.records.append(rec)
                seen_folders.add(folder)

                for alias in (folder, folder.replace("_", " "), sci, common):
                    self._register_alias(alias, folder)
                if common:
                    self._register_alias(f"{common} fish", folder)

        species_prompts: list[str] = []
        for rec in self.records:
            if rec.common_name:
                species_prompts.append(rec.common_name)
                self.prompt_to_folder[rec.common_name] = rec.folder
            elif rec.scientific_name:
                species_prompts.append(rec.scientific_name)
                self.prompt_to_folder[rec.scientific_name] = rec.folder
            else:
                label = rec.folder.replace("_", " ")
                species_prompts.append(label)
                self.prompt_to_folder[label] = rec.folder

        self.prompts = list(GENERIC_MARINE_PROMPTS)
        for p in species_prompts:
            if p not in self.prompts:
                self.prompts.append(p)

        logger.info(
            "Species catalog: %d species, %d YOLO-World prompts (%d generic + %d species)",
            len(self.records),
            len(self.prompts),
            len(GENERIC_MARINE_PROMPTS),
            len(species_prompts),
        )

    def _dedupe_prompts(self, names: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for name in names:
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(name.strip())
        return out

    def _generic_open_vocab_prompts(self, folder_class: str | None = None) -> list[str]:
        """Fish-only prompts for detection — avoids coral/reef false boxes."""
        priority: list[str] = list(FISH_DETECTION_PROMPTS)
        if folder_class and folder_class != "unknown":
            rec = self.get_by_folder(folder_class)
            extra = [
                folder_class.replace("_", " "),
                f"{folder_class.replace('_', ' ')} fish",
            ]
            if rec:
                if rec.common_name:
                    extra.insert(0, rec.common_name)
                if rec.scientific_name:
                    extra.insert(0, rec.scientific_name)
            priority = extra + priority
        return self._dedupe_prompts(priority)

    def _full_open_vocab_prompts(self, folder_class: str | None = None) -> list[str]:
        if not folder_class or folder_class == "unknown":
            return self.prompts
        rec = self.get_by_folder(folder_class)
        priority: list[str] = []
        if rec:
            if rec.common_name:
                priority.append(rec.common_name)
            if rec.scientific_name:
                priority.append(rec.scientific_name)
        priority.extend([
            folder_class.replace("_", " "),
            folder_class,
            f"{folder_class.replace('_', ' ')} fish",
        ])
        return self._dedupe_prompts(priority + self.prompts)

    def open_vocab_classes(self, folder_class: str | None = None) -> list[str]:
        mode = (settings.open_vocab_prompt_mode or "generic").lower()
        if mode == "full":
            return self._full_open_vocab_prompts(folder_class)
        return self._generic_open_vocab_prompts(folder_class)

    def get_by_folder(self, folder: str) -> SpeciesRecord | None:
        key = folder.strip().lower()
        for rec in self.records:
            if rec.folder.lower() == key:
                return rec
        return None

    def resolve_label(self, detected: str) -> str:
        """Map YOLO-World / model text to canonical species_folder slug."""
        if not detected:
            return "unknown"
        key = detected.strip().lower()
        if key in self._alias_to_folder:
            return self._alias_to_folder[key]
        slug = detected.strip().replace(" ", "_").lower()
        if slug in self._alias_to_folder:
            return self._alias_to_folder[slug]
        for rec in self.records:
            if rec.folder.lower() == slug:
                return rec.folder
        return detected.strip()


@lru_cache(maxsize=1)
def get_catalog() -> SpeciesCatalog:
    return SpeciesCatalog()


def reload_catalog() -> SpeciesCatalog:
    get_catalog.cache_clear()
    return get_catalog()
