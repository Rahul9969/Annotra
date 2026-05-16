"""Google Drive folder parsing, listing, and media download."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Read + create/update files in folders the user can access (annotation sidecars).
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

_IMAGE_MIME_PREFIX = "image/"
_IMAGE_NAMES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _q_lit(s: str) -> str:
    """Drive API query literal for name = '...'."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def extract_folder_id(url_or_id: str) -> str:
    """Accept raw folder ID or common shared-folder URL shapes."""
    s = (url_or_id or "").strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{25,}", s):
        return s
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    m = re.search(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)
    raise ValueError("Could not parse a Google Drive folder ID from the link. Paste a shared-folder URL or the raw folder ID.")


def _is_image_file(meta: dict[str, Any]) -> bool:
    mime = (meta.get("mimeType") or "").lower()
    if mime.startswith(_IMAGE_MIME_PREFIX):
        return True
    name = (meta.get("name") or "").lower()
    dot = name.rfind(".")
    if dot < 0:
        return False
    return name[dot:] in _IMAGE_NAMES


def drive_service(access_token: str):
    creds = Credentials(token=access_token, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_image_files_recursive(service, root_folder_id: str) -> list[dict[str, str]]:
    """
    Walk Drive folder tree. Each entry: rel_path (posix), drive_file_id, virtual_path (same as rel_path).
    Skips subtree `.marine-studio` (annotation sidecar storage).
    """
    results: list[dict[str, str]] = []

    def walk(folder_id: str, prefix: str) -> None:
        q = f"'{folder_id}' in parents and trashed = false"
        page_token: str | None = None
        while True:
            resp = (
                service.files()
                .list(
                    q=q,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    pageSize=200,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for f in resp.get("files", []):
                name = f.get("name") or ""
                mid = f.get("id") or ""
                mime = f.get("mimeType") or ""
                if mime == "application/vnd.google-apps.folder":
                    if name == ".marine-studio":
                        continue
                    walk(mid, f"{prefix}{name}/")
                    continue
                if _is_image_file(f):
                    rel = f"{prefix}{name}".replace("\\", "/")
                    results.append(
                        {
                            "rel_path": rel,
                            "path": rel,
                            "drive_file_id": mid,
                            "folder": Path(rel).parent.as_posix() if "/" in rel else "",
                        }
                    )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    walk(root_folder_id, "")
    results.sort(key=lambda x: x["rel_path"].lower())
    return results


def get_file_metadata(service, file_id: str) -> dict[str, Any]:
    return (
        service.files()
        .get(fileId=file_id, fields="id,name,mimeType,modifiedTime,size", supportsAllDrives=True)
        .execute()
    )


def list_files_in_folder(
    service,
    folder_id: str,
    *,
    mime_contains: str | None = None,
) -> list[dict[str, str]]:
    """Non-recursive file list in a folder."""
    q = f"'{folder_id}' in parents and trashed = false"
    if mime_contains:
        q += f" and mimeType contains '{mime_contains}'"
    out: list[dict[str, str]] = []
    page_token: str | None = None
    while True:
        resp = (
            service.files()
            .list(
                q=q,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                pageToken=page_token,
                pageSize=200,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for f in resp.get("files", []):
            if f.get("mimeType") == "application/vnd.google-apps.folder":
                continue
            out.append(
                {
                    "id": f.get("id") or "",
                    "name": f.get("name") or "",
                    "modifiedTime": f.get("modifiedTime") or "",
                }
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def download_file_bytes(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def find_child_folder(service, parent_id: str, name: str) -> str | None:
    q = (
        f"name = {_q_lit(name)} and '{parent_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    resp = (
        service.files()
        .list(
            q=q,
            spaces="drive",
            fields="files(id, name)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def create_folder(service, parent_id: str, name: str) -> str:
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    f = (
        service.files()
        .create(body=body, fields="id", supportsAllDrives=True)
        .execute()
    )
    return f["id"]


def ensure_child_folder(service, parent_id: str, name: str) -> str:
    found = find_child_folder(service, parent_id, name)
    if found:
        return found
    return create_folder(service, parent_id, name)


def find_file_in_folder(service, parent_id: str, file_name: str) -> str | None:
    q = f"name = {_q_lit(file_name)} and '{parent_id}' in parents and trashed = false"
    resp = (
        service.files()
        .list(
            q=q,
            spaces="drive",
            fields="files(id, name)",
            pageSize=5,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def upload_or_update_json(service, parent_id: str, file_name: str, body_json: bytes, existing_file_id: str | None) -> str:
    from googleapiclient.http import MediaInMemoryUpload

    media = MediaInMemoryUpload(body_json, mimetype="application/json", resumable=True)
    if existing_file_id:
        service.files().update(
            fileId=existing_file_id,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return existing_file_id
    meta = {"name": file_name, "parents": [parent_id]}
    f = (
        service.files()
        .create(body=meta, media_body=media, fields="id", supportsAllDrives=True)
        .execute()
    )
    return f["id"]
