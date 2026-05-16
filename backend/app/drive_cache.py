"""Local disk cache for Google Drive images (faster repeat loads + batch inference)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import settings
from app.drive_client import download_file_bytes, drive_service, get_file_metadata


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)[:120]


def cache_root() -> Path:
    root = settings.drive_cache_dir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _meta_path(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".meta.json")


def _cache_file(project_id: int, drive_file_id: str, rel_path: str) -> Path:
    ext = Path(rel_path).suffix.lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        ext = ".jpg"
    return cache_root() / str(project_id) / f"{drive_file_id}{ext}"


def ensure_drive_image_cached(
    project_id: int,
    drive_file_id: str,
    rel_path: str,
    access_token: str,
    *,
    force: bool = False,
) -> Path:
    """
    Return path to a local cached copy. Re-downloads when Drive modifiedTime changes.
    """
    dest = _cache_file(project_id, drive_file_id, rel_path)
    if not settings.drive_cache_enabled:
        force = True

    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = _meta_path(dest)

    svc = drive_service(access_token)
    meta = get_file_metadata(svc, drive_file_id)
    drive_mtime = meta.get("modifiedTime") or ""

    if not force and dest.is_file() and meta_path.is_file():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if stored.get("modifiedTime") == drive_mtime and stored.get("drive_file_id") == drive_file_id:
                return dest
        except (json.JSONDecodeError, OSError):
            pass

    raw = download_file_bytes(svc, drive_file_id)
    dest.write_bytes(raw)
    meta_path.write_text(
        json.dumps(
            {
                "drive_file_id": drive_file_id,
                "rel_path": rel_path,
                "modifiedTime": drive_mtime,
                "size": len(raw),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return dest


def clear_project_cache(project_id: int) -> int:
    """Remove cached files for a project. Returns bytes freed (approx)."""
    folder = cache_root() / str(project_id)
    if not folder.is_dir():
        return 0
    total = 0
    for p in folder.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
                p.unlink()
            except OSError:
                pass
    try:
        folder.rmdir()
    except OSError:
        pass
    return total


def cache_stats(project_id: int | None = None) -> dict:
    root = cache_root()
    if project_id is not None:
        folders = [root / str(project_id)]
    else:
        folders = [p for p in root.iterdir() if p.is_dir()]
    files = 0
    bytes_total = 0
    for folder in folders:
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.is_file() and not p.name.endswith(".meta.json"):
                files += 1
                bytes_total += p.stat().st_size
    return {"files": files, "bytes": bytes_total}
