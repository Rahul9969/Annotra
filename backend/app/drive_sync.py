"""Write / import annotation JSON sidecars in Google Drive (.marine-studio/annotations)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.database import Annotation, ImageRecord, Project
from app.drive_client import (
    download_file_bytes,
    drive_service,
    ensure_child_folder,
    find_child_folder,
    find_file_in_folder,
    list_files_in_folder,
    upload_or_update_json,
)


def _annotation_to_dict(a: Annotation) -> dict:
    return {
        "id": a.id,
        "class_id": a.class_id,
        "class_name": a.class_name,
        "confidence": a.confidence,
        "x": a.x,
        "y": a.y,
        "w": a.w,
        "h": a.h,
        "rotation": a.rotation,
        "polygon": a.polygon,
        "attributes": a.attributes or {},
        "source": a.source,
        "z_index": a.z_index,
        "locked": a.locked,
        "hidden": a.hidden,
    }


def sync_image_annotations_to_drive(
    db: Session,
    project: Project,
    image: ImageRecord,
    access_token: str,
) -> None:
    if project.source != "drive" or not project.drive_folder_id:
        return
    if not image.drive_file_id:
        return

    rows = db.query(Annotation).filter(Annotation.image_id == image.id).order_by(Annotation.z_index).all()
    payload = {
        "schema": "marine-studio-annotations/v1",
        "image_id": image.id,
        "drive_file_id": image.drive_file_id,
        "rel_path": image.rel_path or image.path,
        "status": image.status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "annotations": [_annotation_to_dict(a) for a in rows],
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    service = drive_service(access_token)
    root = project.drive_folder_id
    marine = ensure_child_folder(service, root, ".marine-studio")
    ann_dir = ensure_child_folder(service, marine, "annotations")
    fname = f"{image.drive_file_id}.json"
    existing_id = find_file_in_folder(service, ann_dir, fname)
    upload_or_update_json(service, ann_dir, fname, raw, existing_id)


def _parse_sidecar(raw: bytes) -> dict | None:
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != "marine-studio-annotations/v1":
        return None
    return data


def _apply_sidecar_to_image(db: Session, img: ImageRecord, data: dict, *, overwrite: bool) -> bool:
    anns_data = data.get("annotations")
    if not isinstance(anns_data, list):
        return False

    db_count = db.query(Annotation).filter(Annotation.image_id == img.id).count()
    sidecar_count = len(anns_data)

    sidecar_updated: datetime | None = None
    if data.get("updated_at"):
        try:
            sidecar_updated = datetime.fromisoformat(str(data["updated_at"]).replace("Z", "+00:00"))
        except ValueError:
            sidecar_updated = None

    should = overwrite
    if not should:
        if db_count == 0 and sidecar_count > 0:
            should = True
        elif sidecar_updated and img.annotated_at:
            ann_at = img.annotated_at
            if ann_at.tzinfo is None:
                ann_at = ann_at.replace(tzinfo=timezone.utc)
            should = sidecar_updated > ann_at
        elif sidecar_count > db_count:
            should = True

    if not should:
        return False

    db.query(Annotation).filter(Annotation.image_id == img.id).delete(synchronize_session=False)
    for i, a in enumerate(anns_data):
        if not isinstance(a, dict):
            continue
        db.add(
            Annotation(
                image_id=img.id,
                class_id=int(a.get("class_id") or 0),
                class_name=str(a.get("class_name") or "unknown"),
                confidence=float(a.get("confidence") or 1.0),
                x=float(a.get("x") or 0),
                y=float(a.get("y") or 0),
                w=float(a.get("w") or 0),
                h=float(a.get("h") or 0),
                rotation=float(a.get("rotation") or 0),
                polygon=a.get("polygon"),
                attributes=a.get("attributes") or {},
                source=str(a.get("source") or "human"),
                z_index=int(a.get("z_index") or i),
                locked=bool(a.get("locked")),
                hidden=bool(a.get("hidden")),
            )
        )
    status = data.get("status")
    if status in ("unannotated", "ai", "verified", "flagged"):
        img.status = status
    elif sidecar_count > 0:
        img.status = "ai"
    img.annotated_at = sidecar_updated or datetime.now(timezone.utc)
    return True


def import_annotations_from_drive(
    db: Session,
    project: Project,
    access_token: str,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """
    Load JSON sidecars from Drive folder `.marine-studio/annotations/` into the DB.
    Matches images by drive_file_id (filename or JSON field).
    """
    if project.source != "drive" or not project.drive_folder_id:
        return {"imported": 0, "skipped": 0, "files": 0}

    service = drive_service(access_token)
    marine = find_child_folder(service, project.drive_folder_id, ".marine-studio")
    if not marine:
        return {"imported": 0, "skipped": 0, "files": 0}
    ann_dir = find_child_folder(service, marine, "annotations")
    if not ann_dir:
        return {"imported": 0, "skipped": 0, "files": 0}

    json_files = [f for f in list_files_in_folder(service, ann_dir) if f["name"].endswith(".json")]
    by_drive_id = {
        r.drive_file_id: r
        for r in db.query(ImageRecord).filter(ImageRecord.project_id == project.id).all()
        if r.drive_file_id
    }

    imported = 0
    skipped = 0
    for jf in json_files:
        fid = jf["id"]
        try:
            raw = download_file_bytes(service, fid)
        except Exception:
            skipped += 1
            continue
        data = _parse_sidecar(raw)
        if not data:
            skipped += 1
            continue

        drive_file_id = data.get("drive_file_id") or Path(jf["name"]).stem
        img = by_drive_id.get(drive_file_id)
        if not img:
            skipped += 1
            continue

        if _apply_sidecar_to_image(db, img, data, overwrite=overwrite):
            imported += 1
        else:
            skipped += 1

    db.commit()
    return {"imported": imported, "skipped": skipped, "files": len(json_files)}
