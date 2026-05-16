"""Consistent filesystem paths across Windows/Linux."""
from __future__ import annotations

import os
from pathlib import Path


def norm_path(p: str) -> str:
    """Resolve and normalize so the same folder always maps to one DB project."""
    p = (p or "").strip()
    if p.lower().startswith("gdrive:"):
        rest = p.split(":", 1)[1].strip()
        return f"gdrive:{rest}"
    try:
        resolved = str(Path(p).expanduser().resolve())
    except OSError:
        resolved = str(Path(p).expanduser())
    return os.path.normcase(resolved)
