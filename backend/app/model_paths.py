"""Resolve model weights: backend folder first (fast local load), then models_dir, then download."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def backend_models_dir() -> Path:
    return _BACKEND_DIR


def resolve_model_path(
    name: str | Path | None,
    *,
    min_bytes: int = 500_000,
) -> Path | None:
    """
    Find a model file without redundant downloads.
    Order: absolute path → backend/*.pt → MARINE_* env path → models_dir → None (caller may download by name).
    """
    if not name:
        return None
    p = Path(name).expanduser()
    if p.is_file() and p.stat().st_size >= min_bytes:
        return p.resolve()

    basename = p.name
    models_dir = _BACKEND_DIR
    try:
        from app.config import settings as cfg

        models_dir = cfg.models_dir
    except Exception:
        pass
    for candidate in (
        _BACKEND_DIR / basename,
        models_dir / basename,
        p if p.is_absolute() else None,
    ):
        if candidate is None:
            continue
        if candidate.is_file() and candidate.stat().st_size >= min_bytes:
            logger.info("Using local weights: %s", candidate)
            return candidate.resolve()

    return None


def resolve_model_path_or_name(name: str | Path) -> str:
    """Path for YOLO/SAM loader, or original name to trigger ultralytics hub download."""
    resolved = resolve_model_path(name)
    return str(resolved) if resolved else str(name)
