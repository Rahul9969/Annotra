import asyncio
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.collaboration import broadcast_project, register_ws, unregister_ws
from app.database import (
    Annotation,
    ClassLabel,
    HistoryEntry,
    ImageRecord,
    Project,
    get_db,
    seed_project_classes,
)
from app.database import CLASS_COLORS
from app.batch import batch_annotator, run_batch_job
from app.export_engine import _collect_subfolder_classes, export_dataset
from app.import_engine import import_dataset
from app.drive_client import (
    DRIVE_SCOPES,
    download_file_bytes,
    drive_service,
    extract_folder_id,
    list_image_files_recursive,
)
from app.drive_cache import cache_stats, clear_project_cache
from app.drive_sync import import_annotations_from_drive, sync_image_annotations_to_drive
from app.media_resolver import get_project_local_mirror, resolve_image_path
from app.model_ensemble import ensemble
from app.pipeline import registry, run_pipeline
from app.schemas import (
    AISettings,
    AnnotateRequest,
    AnnotateResponse,
    AnnotationOut,
    MagicSegmentRequest,
    SegmentResponse,
    SmartSegmentRequest,
    BatchStartRequest,
    DriveOAuthRefresh,
    ExportRequest,
    ImportDatasetRequest,
    LocalMirrorBody,
    ProjectCreate,
    ProjectDriveCreate,
    ProjectOut,
    ProjectShareLookup,
    RenameClassBody,
    SaveAnnotationsRequest,
    ShareTokenResponse,
    StatsOut,
    TrainRequest,
)
from app.paths import norm_path
from app.species import discover_species_classes, is_valid_species_folder, species_from_image_path

router = APIRouter()

_oauth_lock = threading.Lock()
_oauth_results: dict[str, dict] = {}


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:].strip() or None


def _google_oauth_client_config() -> dict:
    from app.config import settings

    uri = settings.google_oauth_redirect_uri
    return {
        "web": {
            "client_id": settings.google_client_id or "",
            "client_secret": settings.google_client_secret or "",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [uri],
        }
    }


def _maybe_sync_drive(
    db: Session,
    project_id: int,
    image_id: int,
    authorization: str | None,
) -> None:
    token = _bearer_token(authorization)
    if not token:
        return
    proj = db.query(Project).filter(Project.id == project_id).first()
    img = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if proj and img and getattr(proj, "source", "local") == "drive":
        sync_image_annotations_to_drive(db, proj, img, token)


def _norm_local_mirror(p: str) -> str:
    return os.path.normcase(str(Path(p).expanduser().resolve()))


def _find_project_by_root(db: Session, root_path: str) -> Project | None:
    target = norm_path(root_path)
    for proj in db.query(Project).all():
        if norm_path(proj.root_path) == target:
            return proj
    return None


@router.api_route("/health", methods=["GET", "HEAD"])
def health():
    from app.config import settings as cfg
    from app.segmentation import sam_status
    from app.species_catalog import get_catalog

    sam_st, sam_err = sam_status()
    catalog = get_catalog()

    # Grounding DINO + SAM2 status
    gdino_st, gdino_err = "disabled", None
    sam2_st, sam2_err = "disabled", None
    if cfg.enable_grounding_dino:
        try:
            from app.grounding_dino import gdino_status

            gdino_st, gdino_err = gdino_status()
        except Exception:
            gdino_st = "not_loaded"
    if cfg.enable_sam2:
        try:
            from app.sam2_refiner import sam2_status as _sam2_status

            sam2_st, sam2_err = _sam2_status()
        except Exception:
            sam2_st = "not_loaded"

    return {
        "status": "ok",
        "device": ensemble.device,
        "yolo_status": ensemble.model_status,
        "yolo_error": ensemble.model_error,
        "models_loaded": ensemble.loaded_sources,
        "species_in_catalog": len(catalog.records),
        "open_vocab_prompts": len(catalog.prompts),
        "weights_custom": cfg.custom_yolo,
        "tflite_models": cfg.tflite_models,
        "open_vocab": cfg.enable_open_vocab,
        "open_vocab_model": cfg.open_vocab_model,
        "using_custom_weights": ensemble.using_custom_weights,
        "fast_folder_mode": cfg.fast_folder_mode,
        "auto_use_all_models": cfg.auto_use_all_models,
        "folder_label_mode": cfg.folder_label_mode,
        "enable_open_vocab": cfg.enable_open_vocab,
        "open_vocab_prompt_mode": cfg.open_vocab_prompt_mode,
        "open_vocab_prompt_count": len(catalog.open_vocab_classes()),
        "batch_fast_mode": cfg.batch_fast_mode,
        "batch_use_open_vocab": cfg.batch_use_open_vocab,
        "batch_imgsz": cfg.batch_imgsz,
        "batch_open_vocab_imgsz": cfg.batch_open_vocab_imgsz,
        "batch_inference_size": cfg.batch_inference_size,
        "google_drive_oauth_configured": bool(cfg.google_client_id and cfg.google_client_secret),
        "drive_cache_enabled": cfg.drive_cache_enabled,
        "sam_segment_enabled": cfg.enable_sam_segment,
        "sam_status": sam_st,
        "sam_error": sam_err,
        "grounding_dino_enabled": cfg.enable_grounding_dino,
        "grounding_dino_status": gdino_st,
        "grounding_dino_error": gdino_err,
        "sam2_enabled": cfg.enable_sam2,
        "sam2_status": sam2_st,
        "sam2_error": sam2_err,
        "sam2_refine_boxes": cfg.sam2_refine_boxes,
        "weights_dir": str(__import__("app.model_paths", fromlist=["backend_models_dir"]).backend_models_dir()),
        "custom_yolo_resolved": cfg.custom_yolo,
        "sam_model_resolved": cfg.sam_model,
        "open_vocab_resolved": cfg.open_vocab_model,
    }


@router.get("/drive/oauth/start")
def drive_oauth_start():
    """Returns URL to open in a browser (must match GCP OAuth redirect)."""
    from google_auth_oauthlib.flow import Flow

    from app.config import settings

    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            501,
            "Set MARINE_GOOGLE_CLIENT_ID and MARINE_GOOGLE_CLIENT_SECRET in backend .env "
            "(OAuth 2.0 Web client, redirect = MARINE_GOOGLE_OAUTH_REDIRECT_URI).",
        )
    state = str(uuid.uuid4())
    flow = Flow.from_client_config(
        _google_oauth_client_config(),
        scopes=DRIVE_SCOPES,
        redirect_uri=settings.google_oauth_redirect_uri,
    )
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        state=state,
    )
    return {"auth_url": auth_url, "state": state}


@router.get("/drive/oauth/callback")
def drive_oauth_callback(code: str, state: str):
    from google_auth_oauthlib.flow import Flow

    from app.config import settings

    flow = Flow.from_client_config(
        _google_oauth_client_config(),
        scopes=DRIVE_SCOPES,
        redirect_uri=settings.google_oauth_redirect_uri,
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    payload = {
        "access_token": creds.token,
        "refresh_token": getattr(creds, "refresh_token", None),
    }
    if creds.expiry:
        payload["expires_at"] = int(creds.expiry.timestamp())
    with _oauth_lock:
        _oauth_results[state] = payload

    base = settings.google_oauth_frontend_redirect.rstrip("/")
    sep = "&" if "?" in base else "?"
    return RedirectResponse(f"{base}{sep}drive_oauth_state={state}")


@router.get("/drive/oauth/result")
def drive_oauth_result(state: str):
    with _oauth_lock:
        data = _oauth_results.pop(state, None)
    if not data:
        raise HTTPException(
            404,
            "OAuth result missing or already consumed. Start Connect Google again.",
        )
    return data


@router.post("/drive/oauth/refresh")
def drive_oauth_refresh(body: DriveOAuthRefresh):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    from app.config import settings

    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(501, "Google OAuth not configured.")
    creds = Credentials(
        token=None,
        refresh_token=body.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
    )
    creds.refresh(Request())
    out: dict = {"access_token": creds.token}
    if creds.expiry:
        out["expires_at"] = int(creds.expiry.timestamp())
    return out


def _index_drive_files_into_project(db: Session, proj: Project, files: list[dict]) -> None:
    scanned = {f["path"] for f in files}
    for row in db.query(ImageRecord).filter(ImageRecord.project_id == proj.id).all():
        if row.path not in scanned:
            db.query(Annotation).filter(Annotation.image_id == row.id).delete(
                synchronize_session=False,
            )
            db.delete(row)
    existing_by_path = {
        r.path: r for r in db.query(ImageRecord).filter(ImageRecord.project_id == proj.id).all()
    }
    for f in files:
        path = f["path"]
        fid = f["drive_file_id"]
        rel = f.get("rel_path") or path
        if path in existing_by_path:
            r = existing_by_path[path]
            r.drive_file_id = fid
            sp = species_from_image_path(path, proj.root_path)
            if sp != "unknown":
                r.species_class = sp
            continue
        sp = species_from_image_path(path, proj.root_path)
        db.add(
            ImageRecord(
                project_id=proj.id,
                path=path,
                rel_path=rel,
                drive_file_id=fid,
                species_class=sp,
                status="unannotated",
            ),
        )
    db.commit()
    folder_classes = discover_species_classes(files, proj.root_path)
    if folder_classes:
        seed_project_classes(db, proj.id, folder_classes, folder_names_only=True, replace=True)


@router.post("/projects/from-drive", response_model=ProjectOut)
def create_project_from_drive(body: ProjectDriveCreate, db: Session = Depends(get_db)):
    from app.config import settings

    if not settings.google_client_id:
        raise HTTPException(501, "Google Drive is not configured on the server.")
    folder_id = extract_folder_id(body.folder_url)
    try:
        svc = drive_service(body.access_token)
        svc.files().get(fileId=folder_id, fields="id", supportsAllDrives=True).execute()
    except Exception as e:
        raise HTTPException(401, f"Google Drive access failed: {e}") from None

    files = list_image_files_recursive(svc, folder_id)
    if not files:
        raise HTTPException(400, "No image files found in that Drive folder.")

    root_path = f"gdrive:{folder_id}"
    existing = db.query(Project).filter(Project.root_path == root_path).first()
    if existing:
        existing.name = body.name
        existing.source = "drive"
        existing.drive_folder_id = folder_id
        db.commit()
        db.refresh(existing)
        proj = existing
    else:
        proj = Project(
            name=body.name,
            root_path=root_path,
            source="drive",
            drive_folder_id=folder_id,
        )
        db.add(proj)
        db.commit()
        db.refresh(proj)
        seed_project_classes(db, proj.id, None, folder_names_only=False, replace=False)

    _index_drive_files_into_project(db, proj, files)
    try:
        import_annotations_from_drive(db, proj, body.access_token, overwrite=False)
    except Exception:
        pass
    return _project_out(db, proj)


@router.post("/projects/{project_id}/drive/import-annotations")
def drive_import_annotations(
    project_id: int,
    overwrite: bool = False,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(401, "Authorization: Bearer <google access token> required.")
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj or getattr(proj, "source", "local") != "drive":
        raise HTTPException(400, "Not a Google Drive project.")
    try:
        return import_annotations_from_drive(db, proj, token, overwrite=overwrite)
    except Exception as e:
        raise HTTPException(502, f"Import from Drive failed: {e}") from None


@router.post("/projects/{project_id}/drive/cache/clear")
def drive_clear_cache(project_id: int, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    freed = clear_project_cache(project_id)
    return {"ok": True, "bytes_freed": freed}


@router.get("/projects/{project_id}/drive/cache/stats")
def drive_cache_stats(project_id: int, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    return cache_stats(project_id)


@router.post("/projects/{project_id}/drive/reindex", response_model=ProjectOut)
def reindex_drive_project(
    project_id: int,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    token = _bearer_token(authorization)
    if not token:
        raise HTTPException(401, "Authorization: Bearer <google access token> required.")
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj or getattr(proj, "source", "local") != "drive" or not proj.drive_folder_id:
        raise HTTPException(400, "Not a Google Drive-backed project.")
    try:
        svc = drive_service(token)
        files = list_image_files_recursive(svc, proj.drive_folder_id)
    except Exception as e:
        raise HTTPException(502, f"Drive listing failed: {e}") from None
    if not files:
        raise HTTPException(400, "No images found after re-index.")
    _index_drive_files_into_project(db, proj, files)
    try:
        import_annotations_from_drive(db, proj, token, overwrite=False)
    except Exception:
        pass
    return _project_out(db, proj)


@router.post("/projects/import-dataset")
def import_dataset_project(body: ImportDatasetRequest, db: Session = Depends(get_db)):
    """Import YOLO, COCO, VOC, CSV, or Annotra export folder into an editable project."""
    path = Path(body.dataset_path)
    if not path.is_dir():
        raise HTTPException(400, f"Not a directory: {body.dataset_path}")
    try:
        return import_dataset(db, path, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except OSError as e:
        raise HTTPException(400, f"Cannot read dataset: {e}") from e


@router.post("/projects", response_model=ProjectOut)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    root = norm_path(body.root_path)
    existing = _find_project_by_root(db, root)
    if existing:
        existing.name = body.name
        existing.root_path = root
        db.commit()
        db.refresh(existing)
        return _project_out(db, existing)
    proj = Project(name=body.name, root_path=root)
    db.add(proj)
    db.commit()
    db.refresh(proj)
    folder_only = bool(body.class_names)
    seed_project_classes(db, proj.id, body.class_names, folder_names_only=folder_only, replace=True)
    return _project_out(db, proj)


def _project_settings(proj: Project) -> dict:
    data = proj.settings_json
    return dict(data) if isinstance(data, dict) else {}


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    rows = db.query(Project).order_by(Project.created_at.desc()).all()
    return [_project_out(db, p) for p in rows]


@router.post("/projects/{project_id}/share", response_model=ShareTokenResponse)
def create_or_get_share_token(project_id: int, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    s = _project_settings(proj)
    if not s.get("share_token"):
        s["share_token"] = str(uuid.uuid4())
    proj.settings_json = s
    db.commit()
    return ShareTokenResponse(share_token=s["share_token"])


@router.get("/projects/by-share/{token}", response_model=ProjectShareLookup)
def lookup_project_by_share(token: str, db: Session = Depends(get_db)):
    for proj in db.query(Project).all():
        s = _project_settings(proj)
        if s.get("share_token") == token:
            return ProjectShareLookup(project=_project_out(db, proj), share_token=token)
    raise HTTPException(404, "Invalid or expired share link")


async def _notify_annotation_saved(project_id: int, image_id: int):
    await broadcast_project(
        project_id,
        {"type": "annotations_saved", "image_id": image_id},
    )


@router.websocket("/ws/project/{project_id}")
async def project_collaboration_ws(websocket: WebSocket, project_id: int):
    await websocket.accept()
    await register_ws(project_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await unregister_ws(project_id, websocket)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    return _project_out(db, proj)


@router.delete("/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")

    image_ids = [
        row.id
        for row in db.query(ImageRecord.id).filter(ImageRecord.project_id == project_id).all()
    ]
    if image_ids:
        db.query(Annotation).filter(Annotation.image_id.in_(image_ids)).delete(
            synchronize_session=False
        )
        db.query(HistoryEntry).filter(HistoryEntry.image_id.in_(image_ids)).delete(
            synchronize_session=False
        )
    db.query(ImageRecord).filter(ImageRecord.project_id == project_id).delete(
        synchronize_session=False
    )
    db.query(ClassLabel).filter(ClassLabel.project_id == project_id).delete(
        synchronize_session=False
    )

    if getattr(proj, "source", "local") == "drive":
        try:
            clear_project_cache(project_id)
        except Exception:
            pass

    db.delete(proj)
    db.commit()
    return {"ok": True, "deleted_project_id": project_id}


def _project_out(db: Session, proj: Project) -> ProjectOut:
    total = db.query(ImageRecord).filter(ImageRecord.project_id == proj.id).count()
    annotated = (
        db.query(ImageRecord)
        .filter(ImageRecord.project_id == proj.id, ImageRecord.status.in_(["ai", "verified"]))
        .count()
    )
    return ProjectOut(
        id=proj.id,
        name=proj.name,
        root_path=proj.root_path,
        image_count=total,
        annotated_count=annotated,
        source=getattr(proj, "source", None) or "local",
        drive_folder_id=getattr(proj, "drive_folder_id", None),
        local_mirror_path=get_project_local_mirror(proj),
    )


@router.put("/projects/{project_id}/local-mirror")
def set_project_local_mirror(
    project_id: int,
    body: LocalMirrorBody,
    db: Session = Depends(get_db),
):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    s = _project_settings(proj)
    if body.path and body.path.strip():
        try:
            s["local_mirror_path"] = _norm_local_mirror(body.path.strip())
        except OSError as e:
            raise HTTPException(400, f"Invalid folder path: {e}") from e
    else:
        s.pop("local_mirror_path", None)
    proj.settings_json = s
    db.commit()
    return {"local_mirror_path": s.get("local_mirror_path")}


@router.post("/projects/{project_id}/index")
def index_images(project_id: int, files: list[dict], db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    if getattr(proj, "source", "local") == "drive":
        raise HTTPException(
            400,
            "This project is linked to Google Drive. Use “Refresh Drive index” from the app instead of local folder indexing.",
        )
    root = Path(proj.root_path)
    scanned = {norm_path(f.get("path", "")) for f in files if f.get("path")}
    removed = 0
    for row in db.query(ImageRecord).filter(ImageRecord.project_id == project_id).all():
        if norm_path(row.path) not in scanned:
            db.query(Annotation).filter(Annotation.image_id == row.id).delete(
                synchronize_session=False
            )
            db.delete(row)
            removed += 1

    existing_by_norm = {
        norm_path(r.path): r
        for r in db.query(ImageRecord).filter(ImageRecord.project_id == project_id).all()
    }

    added = 0
    for f in files:
        path = f.get("path", "")
        if not path:
            continue
        exists = existing_by_norm.get(norm_path(path))
        if exists:
            sp = species_from_image_path(path, proj.root_path)
            if sp != "unknown":
                exists.species_class = sp
            continue
        try:
            rel = str(Path(path).relative_to(root))
        except ValueError:
            rel = Path(path).name
        species = species_from_image_path(path, proj.root_path)
        db.add(
            ImageRecord(
                project_id=project_id,
                path=path,
                rel_path=rel,
                species_class=species,
                status="unannotated",
            )
        )
        added += 1
    db.commit()
    folder_classes = discover_species_classes(files, proj.root_path)
    if folder_classes:
        seed_project_classes(db, project_id, folder_classes, folder_names_only=True, replace=True)
    return {
        "added": added,
        "removed": removed,
        "total": db.query(ImageRecord).filter(ImageRecord.project_id == project_id).count(),
        "classes_from_folders": folder_classes,
    }


@router.post("/projects/{project_id}/reindex-species")
def reindex_species(project_id: int, db: Session = Depends(get_db)):
    """Backfill species_class from parent folder names for already-indexed images."""
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    rows = db.query(ImageRecord).filter(ImageRecord.project_id == project_id).all()
    names: set[str] = set()
    for r in rows:
        sp = species_from_image_path(r.path, proj.root_path)
        r.species_class = sp
        if sp != "unknown":
            names.add(sp)
    db.commit()
    if names:
        seed_project_classes(db, project_id, sorted(names), folder_names_only=True, replace=True)
    return {"updated": len(rows), "classes": sorted(names)}


@router.get("/projects/{project_id}/images")
def list_images(
    project_id: int,
    offset: int = 0,
    limit: int = 100,
    search: str = "",
    status: str | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(ImageRecord).filter(ImageRecord.project_id == project_id)
    if search:
        q = q.filter(ImageRecord.path.contains(search))
    if status:
        q = q.filter(ImageRecord.status == status)
    total = q.count()
    rows = q.order_by(ImageRecord.id).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id": r.id,
                "path": r.path,
                "rel_path": r.rel_path,
                "drive_file_id": r.drive_file_id
                if getattr(r, "drive_file_id", None)
                else None,
                "species_class": r.species_class,
                "width": r.width,
                "height": r.height,
                "status": r.status,
                "annotation_count": db.query(Annotation).filter(Annotation.image_id == r.id).count(),
            }
            for r in rows
        ],
    }


def _resolve_image_for_inference(
    db: Session,
    img_row: ImageRecord | None,
    body: AnnotateRequest | MagicSegmentRequest | SmartSegmentRequest,
    authorization: str | None,
) -> tuple[str, str | None]:
    """Return (local path, temp_path to delete)."""
    import base64
    import os
    import tempfile

    temp_path: str | None = None
    img_path = body.image_path

    if img_row and getattr(img_row, "drive_file_id", None):
        token = _bearer_token(authorization)
        if not token and not body.image_base64:
            raise HTTPException(
                401,
                "Connect Google Drive (or send image data) to run AI on cloud-hosted images.",
            )
        if token:
            try:
                proj = (
                    db.query(Project).filter(Project.id == img_row.project_id).first()
                    if img_row.project_id
                    else None
                )
                mirror = get_project_local_mirror(proj) if proj else None
                resolved, _ = resolve_image_path(
                    proj,
                    img_row,
                    local_mirror=mirror,
                    drive_token=token,
                )
                if resolved:
                    return str(resolved), None
                raise HTTPException(502, "Could not resolve image from Drive or local mirror.")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(502, f"Image resolve failed: {e}") from None

    if not Path(img_path).exists():
        if body.image_base64:
            try:
                raw = base64.b64decode(body.image_base64)
            except Exception:
                raise HTTPException(400, "Invalid image_base64") from None
            suf = Path(body.image_path).suffix if body.image_path else ".jpg"
            if not suf.startswith("."):
                suf = ".jpg"
            fd, temp_path = tempfile.mkstemp(suffix=suf)
            os.write(fd, raw)
            os.close(fd)
            return temp_path, temp_path
        raise HTTPException(
            404,
            "Image not found on server. Use the desktop app with the shared dataset folder, "
            "or connect Google Drive / send image_base64.",
        )
    return img_path, None


@router.get("/images/{image_id}/media")
def get_image_media(
    image_id: int,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    img = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    proj = db.query(Project).filter(Project.id == img.project_id).first()
    if proj and getattr(proj, "source", "local") == "drive" and img.drive_file_id:
        token = _bearer_token(authorization)
        if not token:
            raise HTTPException(401, "Authorization: Bearer <google access token> required for Drive images.")
        try:
            mirror = get_project_local_mirror(proj)
            resolved, _ = resolve_image_path(
                proj, img, local_mirror=mirror, drive_token=token
            )
            if not resolved:
                raise HTTPException(404, "Image not available (link local folder or check Drive).")
            raw = resolved.read_bytes()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Image load failed: {e}") from None
        ext = Path(img.path or "").suffix.lower()
        mime = "image/jpeg"
        if ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        elif ext == ".gif":
            mime = "image/gif"
        return Response(content=raw, media_type=mime)
    if Path(img.path).exists():
        return Response(content=Path(img.path).read_bytes(), media_type="application/octet-stream")
    raise HTTPException(404, "Image file not available on server.")


@router.get("/images/{image_id}/annotations")
def get_annotations(image_id: int, db: Session = Depends(get_db)):
    anns = db.query(Annotation).filter(Annotation.image_id == image_id).order_by(Annotation.z_index).all()
    return [
        AnnotationOut(
            id=a.id,
            class_id=a.class_id,
            class_name=a.class_name,
            confidence=a.confidence,
            x=a.x,
            y=a.y,
            w=a.w,
            h=a.h,
            rotation=a.rotation,
            polygon=a.polygon,
            attributes=a.attributes or {},
            source=a.source,
            z_index=a.z_index,
            locked=a.locked,
            hidden=a.hidden,
        )
        for a in anns
    ]


def _persist_annotations(db: Session, body: SaveAnnotationsRequest) -> int:
    img = db.query(ImageRecord).filter(ImageRecord.id == body.image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    db.query(Annotation).filter(Annotation.image_id == body.image_id).delete()
    for a in body.annotations:
        db.add(
            Annotation(
                image_id=body.image_id,
                class_id=a.class_id,
                class_name=a.class_name,
                confidence=a.confidence,
                x=a.x,
                y=a.y,
                w=a.w,
                h=a.h,
                rotation=a.rotation,
                polygon=a.polygon,
                attributes=a.attributes,
                source=a.source,
                z_index=a.z_index,
                locked=a.locked,
                hidden=a.hidden,
            )
        )
    if body.status:
        img.status = body.status
    img.annotated_at = datetime.now(timezone.utc)
    db.commit()
    return img.project_id


@router.post("/annotations/save")
def save_annotations(
    body: SaveAnnotationsRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    pid = _persist_annotations(db, body)
    _maybe_sync_drive(db, pid, body.image_id, authorization)
    background_tasks.add_task(_notify_annotation_saved, pid, body.image_id)
    return {"ok": True}


@router.post("/annotate", response_model=AnnotateResponse)
def annotate_image(
    body: AnnotateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    import os

    img_row = None
    if body.image_id:
        img_row = db.query(ImageRecord).filter(ImageRecord.id == body.image_id).first()

    img_path, temp_path = _resolve_image_for_inference(db, img_row, body, authorization)

    if registry.model_status in ("loading", "not_loaded"):
        raise HTTPException(
            503,
            "AI models are still loading. Please wait until they finish, then try Auto again.",
        )
    if registry.model_status == "error":
        raise HTTPException(503, f"AI model failed to load: {registry.model_error}")

    class_candidates = None
    if body.project_id:
        classes = db.query(ClassLabel).filter(ClassLabel.project_id == body.project_id).all()
        class_candidates = [c.name for c in classes]

    project_root: str | None = None
    project_name: str | None = None
    folder_class = None
    proj = None
    if body.project_id:
        proj = db.query(Project).filter(Project.id == body.project_id).first()
        if proj:
            project_root = proj.root_path
            project_name = proj.name
    if img_row and img_row.species_class and img_row.species_class != "unknown":
        folder_class = img_row.species_class
    if not folder_class or folder_class == "unknown":
        folder_class = species_from_image_path(img_path, project_root)
    if folder_class and not is_valid_species_folder(folder_class, project_name):
        folder_class = None

    try:
        anns, w, h, timing = run_pipeline(
            img_path,
            prompts=body.prompts,
            class_candidates=class_candidates,
            folder_class=folder_class if folder_class and folder_class != "unknown" else None,
            use_all_models=True,
            project_root=project_root,
            project_name=project_name,
            ai_settings=AISettings(**body.ai_settings) if body.ai_settings else None,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    finally:
        if temp_path and os.path.isfile(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    if body.image_id:
        if img_row:
            img_row.width = w
            img_row.height = h
        pid = _persist_annotations(
            db,
            SaveAnnotationsRequest(image_id=body.image_id, annotations=anns, status="ai"),
        )
        _maybe_sync_drive(db, pid, body.image_id, authorization)
        background_tasks.add_task(_notify_annotation_saved, pid, body.image_id)

    return AnnotateResponse(annotations=anns, width=w, height=h, timing_ms=timing)


def _segment_to_annotation(seg, class_name: str) -> AnnotationOut:
    return AnnotationOut(
        class_name=class_name,
        confidence=1.0,
        x=seg.x,
        y=seg.y,
        w=seg.w,
        h=seg.h,
        rotation=0,
        polygon=seg.polygon,
        source=seg.source,
    )


@router.post("/segment/magic", response_model=SegmentResponse)
def segment_magic(
    body: MagicSegmentRequest,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    import os

    from app.config import settings as cfg
    from app.segmentation import segment_from_path_magic

    img_row = None
    if body.image_id:
        img_row = db.query(ImageRecord).filter(ImageRecord.id == body.image_id).first()

    img_path, temp_path = _resolve_image_for_inference(db, img_row, body, authorization)
    tol = body.tolerance if body.tolerance is not None else cfg.magic_wand_tolerance
    try:
        seg = segment_from_path_magic(img_path, body.x, body.y, tol)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        if temp_path and os.path.isfile(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return SegmentResponse(
        polygon=seg.polygon,
        x=seg.x,
        y=seg.y,
        w=seg.w,
        h=seg.h,
        width=seg.width,
        height=seg.height,
        source=seg.source,
    )


@router.post("/segment/smart", response_model=SegmentResponse)
def segment_smart(
    body: SmartSegmentRequest,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    import os

    from app.segmentation import segment_from_path_smart

    if not body.points:
        raise HTTPException(400, "At least one point is required")

    img_row = None
    if body.image_id:
        img_row = db.query(ImageRecord).filter(ImageRecord.id == body.image_id).first()

    img_path, temp_path = _resolve_image_for_inference(db, img_row, body, authorization)
    pts = [(float(p[0]), float(p[1])) for p in body.points]
    lbs = [int(l) for l in body.labels]
    if len(lbs) != len(pts):
        lbs = [1] * len(pts)
    try:
        seg = segment_from_path_smart(img_path, pts, lbs)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        if temp_path and os.path.isfile(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return SegmentResponse(
        polygon=seg.polygon,
        x=seg.x,
        y=seg.y,
        w=seg.w,
        h=seg.h,
        width=seg.width,
        height=seg.height,
        source=seg.source,
    )


@router.post("/batch/start")
async def batch_start(
    body: BatchStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    proj = db.query(Project).filter(Project.id == body.project_id).first()
    drive_token: str | None = None
    local_mirror: str | None = get_project_local_mirror(proj) if proj else None
    if proj and getattr(proj, "source", "local") == "drive":
        drive_token = _bearer_token(authorization)
        if not drive_token and not local_mirror:
            raise HTTPException(
                401,
                "Connect Google Drive or link a local copy of the dataset for batch.",
            )
    if registry.model_status in ("loading", "not_loaded"):
        raise HTTPException(
            503,
            "AI models are still loading. Please wait until they finish, then start batch again.",
        )
    if registry.model_status == "error":
        raise HTTPException(503, f"AI model failed to load: {registry.model_error}")

    q = db.query(ImageRecord).filter(ImageRecord.project_id == body.project_id)
    if body.image_ids:
        q = q.filter(ImageRecord.id.in_(body.image_ids))

    if body.skip_annotated:
        rows = q.filter(ImageRecord.status == "unannotated").all()
        seen = {r.id for r in rows}
        for r in db.query(ImageRecord).filter(ImageRecord.project_id == body.project_id).all():
            if r.id in seen:
                continue
            if r.status != "ai":
                continue
            if db.query(Annotation).filter(Annotation.image_id == r.id).count() == 0:
                rows.append(r)
                seen.add(r.id)
    else:
        rows = q.all()

    images = [
        (
            r.id,
            r.path,
            r.species_class or species_from_image_path(r.path, None),
            getattr(r, "drive_file_id", None),
        )
        for r in rows
    ]
    if not images:
        raise HTTPException(
            400,
            "No images to process. Re-open your folder or click Auto on a single image.",
        )
    ai_settings = get_ai_settings()
    if body.pipeline_mode:
        ai_settings.pipeline_mode = body.pipeline_mode

    job_id = batch_annotator.create_job(
        body.project_id,
        images,
        prompts=body.prompts,
        ai_settings=ai_settings,
        drive_access_token=drive_token,
        local_mirror_path=local_mirror,
    )
    job = batch_annotator.get_job(job_id)
    if job:
        job.status = "running"
        job.start_time = time.time()
    background_tasks.add_task(run_batch_job, job_id)
    return {"job_id": job_id, "total": len(images), "uses_local_mirror": bool(local_mirror)}


@router.get("/batch/{job_id}/status")
def batch_status(job_id: str):
    job = batch_annotator.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return batch_annotator.status(job_id)


@router.post("/batch/{job_id}/pause")
def batch_pause(job_id: str):
    batch_annotator.pause(job_id)
    return {"ok": True}


@router.post("/batch/{job_id}/resume")
def batch_resume(job_id: str):
    batch_annotator.resume(job_id)
    return {"ok": True}


@router.post("/batch/{job_id}/cancel")
def batch_cancel(job_id: str):
    batch_annotator.cancel(job_id)
    return {"ok": True}


async def _ws_send_json(websocket: WebSocket, payload: dict) -> bool:
    """Send on batch progress socket; ignore client disconnect."""
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@router.websocket("/ws/batch/{job_id}")
async def batch_ws(websocket: WebSocket, job_id: str):
    """Stream batch progress. Job already runs from POST /batch/start; disconnect does not cancel."""
    await websocket.accept()
    if not batch_annotator.get_job(job_id):
        await websocket.close(code=4004)
        return

    try:
        while True:
            st = batch_annotator.status(job_id)
            if not await _ws_send_json(websocket, st.model_dump()):
                break
            if st.status in ("completed", "cancelled"):
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@router.get("/projects/{project_id}/classes")
def list_classes(project_id: int, db: Session = Depends(get_db)):
    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")
    root = proj.root_path if proj else None
    folder_names = set(_collect_subfolder_classes(db, project_id, root))
    db_rows = (
        db.query(ClassLabel)
        .filter(ClassLabel.project_id == project_id)
        .order_by(ClassLabel.sort_order)
        .all()
    )
    ann_names = {
        row[0]
        for row in db.query(Annotation.class_name)
        .join(ImageRecord, ImageRecord.id == Annotation.image_id)
        .filter(ImageRecord.project_id == project_id)
        .distinct()
        .all()
        if row[0]
    }
    by_name = {c.name: c for c in db_rows}
    names_ordered: list[str] = []
    seen: set[str] = set()
    for n in [c.name for c in db_rows] + sorted(folder_names) + sorted(ann_names):
        if n and n not in seen:
            seen.add(n)
            names_ordered.append(n)
    if not names_ordered:
        names_ordered = ["unknown", "fish"]
    return [
        {
            "id": by_name[n].id if n in by_name else 0,
            "name": n,
            "color": by_name[n].color if n in by_name else CLASS_COLORS[i % len(CLASS_COLORS)],
            "hotkey": by_name[n].hotkey if n in by_name else None,
            "supercategory": by_name[n].supercategory if n in by_name else None,
        }
        for i, n in enumerate(names_ordered)
    ]


@router.patch("/projects/{project_id}/classes/rename")
def rename_class(project_id: int, body: RenameClassBody, db: Session = Depends(get_db)):
    old_name = body.old_name.strip()
    new_name = body.new_name.strip()
    if not old_name or not new_name:
        raise HTTPException(400, "old_name and new_name are required")
    if old_name == new_name:
        return {"ok": True, "renamed": 0}

    proj = db.query(Project).filter(Project.id == project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")

    existing = (
        db.query(ClassLabel)
        .filter(ClassLabel.project_id == project_id, ClassLabel.name == new_name)
        .first()
    )
    if existing and new_name != old_name:
        raise HTTPException(409, f"Class '{new_name}' already exists")

    cls = (
        db.query(ClassLabel)
        .filter(ClassLabel.project_id == project_id, ClassLabel.name == old_name)
        .first()
    )
    if cls:
        cls.name = new_name

    image_ids = [
        row.id
        for row in db.query(ImageRecord.id).filter(ImageRecord.project_id == project_id).all()
    ]
    updated = 0
    if image_ids:
        updated = (
            db.query(Annotation)
            .filter(Annotation.image_id.in_(image_ids), Annotation.class_name == old_name)
            .update({Annotation.class_name: new_name}, synchronize_session=False)
        )
    if not cls and updated == 0:
        raise HTTPException(404, f"Class '{old_name}' not found in this project")

    db.commit()
    return {"ok": True, "old_name": old_name, "new_name": new_name, "annotations_updated": updated}


@router.get("/projects/{project_id}/stats", response_model=StatsOut)
def project_stats(project_id: int, db: Session = Depends(get_db)):
    total = db.query(ImageRecord).filter(ImageRecord.project_id == project_id).count()
    status_rows = (
        db.query(ImageRecord.status, func.count())
        .filter(ImageRecord.project_id == project_id)
        .group_by(ImageRecord.status)
        .all()
    )
    status_counts = {s: c for s, c in status_rows}
    annotated = status_counts.get("ai", 0) + status_counts.get("verified", 0)
    species = (
        db.query(Annotation.class_name, func.count())
        .join(ImageRecord, ImageRecord.id == Annotation.image_id)
        .filter(ImageRecord.project_id == project_id)
        .group_by(Annotation.class_name)
        .all()
    )
    box_count = (
        db.query(func.count(Annotation.id))
        .join(ImageRecord, ImageRecord.id == Annotation.image_id)
        .filter(ImageRecord.project_id == project_id)
        .scalar()
        or 0
    )
    return StatsOut(
        total_images=total,
        annotated_pct=round(100 * annotated / total, 1) if total else 0,
        avg_boxes=round(box_count / max(annotated, 1), 2),
        species_distribution={n: c for n, c in species},
        status_counts=status_counts,
    )


@router.post("/export")
def export(
    body: ExportRequest,
    db: Session = Depends(get_db),
    authorization: str | None = Header(None, alias="Authorization"),
):
    proj = db.query(Project).filter(Project.id == body.project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")

    local_mirror = body.local_mirror_path or get_project_local_mirror(proj)
    drive_token: str | None = None
    if getattr(proj, "source", "local") == "drive":
        drive_token = _bearer_token(authorization)
        if not drive_token and not local_mirror:
            raise HTTPException(
                401,
                "Connect Google Drive or link your local dataset folder to export.",
            )

    try:
        return export_dataset(
            db,
            body.project_id,
            body.output_dir,
            fmt=body.format,
            split_train=body.split_train,
            split_val=body.split_val,
            confidence_min=body.confidence_min,
            reviewed_only=body.reviewed_only,
            class_filter=body.class_filter,
            include_labeled_previews=body.include_labeled_previews,
            local_mirror=local_mirror,
            drive_token=drive_token,
            create_zip=body.create_zip,
        )
    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}") from e


@router.post("/train")
def train_model(body: TrainRequest, db: Session = Depends(get_db)):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise HTTPException(500, "ultralytics not installed")

    proj = db.query(Project).filter(Project.id == body.project_id).first()
    if not proj:
        raise HTTPException(404, "Project not found")

    import tempfile

    tmp = tempfile.mkdtemp(prefix="marine_train_")
    export_dataset(db, body.project_id, tmp, fmt="yolo", reviewed_only=True)
    data_yaml = Path(tmp) / "data.yaml"
    model = YOLO(body.model)
    results = model.train(
        data=str(data_yaml),
        epochs=body.epochs,
        imgsz=body.imgsz,
        batch=body.batch,
        device=registry.device,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        registry.hot_swap_yolo(str(best))
    return {"ok": True, "weights": str(best) if best.exists() else None}


@router.get("/ai/settings")
def get_ai_settings():
    from app.config import settings

    return AISettings(
        confidence=settings.confidence,
        iou_threshold=settings.iou_threshold,
        max_boxes=settings.max_boxes,
        batch_size=settings.batch_size,
        thread_pool_size=settings.thread_pool_size,
        device=settings.device,
        half_precision=settings.half_precision,
        yolo_model=settings.yolo_model,
        custom_yolo=settings.custom_yolo,
        enable_sam=getattr(settings, "enable_sam", settings.enable_sam2),
        annotation_mode=settings.annotation_mode,
    )


@router.put("/ai/settings")
def update_ai_settings(body: AISettings):
    from app import config

    for k, v in body.model_dump().items():
        if hasattr(config.settings, k):
            setattr(config.settings, k, v)
        if k == "enable_sam":
            config.settings.enable_sam2 = v
    return body
