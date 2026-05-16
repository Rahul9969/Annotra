"""Resolve image bytes from local mirror, on-disk path, or Google Drive cache."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from app.database import ImageRecord, Project


class MediaSource(str, Enum):
    LOCAL_MIRROR = "local_mirror"
    LOCAL_PATH = "local_path"
    DRIVE_CACHE = "drive_cache"
    MISSING = "missing"


def local_mirror_file(local_root: str | None, rel_path: str | None) -> Path | None:
    if not local_root or not rel_path:
        return None
    p = Path(local_root) / rel_path.replace("/", os.sep)
    try:
        return p if p.is_file() else None
    except OSError:
        return None


def resolve_image_path(
    project: Project | None,
    img: ImageRecord,
    *,
    local_mirror: str | None = None,
    drive_token: str | None = None,
) -> tuple[Path | None, MediaSource]:
    rel = (img.rel_path or img.path or "").replace("\\", "/")

    hit = local_mirror_file(local_mirror, rel)
    if hit:
        return hit, MediaSource.LOCAL_MIRROR

    if img.path:
        direct = Path(img.path)
        if direct.is_file():
            return direct, MediaSource.LOCAL_PATH

    if (
        project
        and getattr(project, "source", "local") == "drive"
        and getattr(img, "drive_file_id", None)
        and drive_token
    ):
        from app.drive_cache import ensure_drive_image_cached

        try:
            cached = ensure_drive_image_cached(
                project.id,
                img.drive_file_id,
                rel or img.path,
                drive_token,
            )
            return cached, MediaSource.DRIVE_CACHE
        except Exception:
            return None, MediaSource.MISSING

    return None, MediaSource.MISSING


def get_project_local_mirror(project: Project | None) -> str | None:
    if not project:
        return None
    data = project.settings_json
    if not isinstance(data, dict):
        return None
    p = data.get("local_mirror_path")
    return str(p) if p else None
