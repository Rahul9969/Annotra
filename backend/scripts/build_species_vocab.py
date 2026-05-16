#!/usr/bin/env python3
"""Regenerate marine_species_vocab.txt from species_mapping.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.species_catalog import GENERIC_MARINE_PROMPTS, reload_catalog


def main() -> None:
    catalog = reload_catalog()
    out = settings.open_vocab_classes_file
    lines = [
        "# Auto-generated from species_mapping.csv — do not edit by hand",
        f"# Species: {len(catalog.records)}",
        "",
        "# Generic marine",
        *GENERIC_MARINE_PROMPTS,
        "",
        "# Species (common names)",
    ]
    for rec in catalog.records:
        if rec.common_name:
            lines.append(rec.common_name)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {out}")


if __name__ == "__main__":
    main()
