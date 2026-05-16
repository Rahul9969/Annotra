"""Parallel batch annotation with progress callbacks."""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from app.config import settings
from app.database import Annotation, ImageRecord, SessionLocal
from app.pipeline import run_pipeline
from app.species import is_valid_species_folder
from app.schemas import AISettings, BatchStatus

logger = logging.getLogger(__name__)


def _annotate_one(
    image_path: str,
    prompts: list[str] | None,
    cfg_dict: dict,
    folder_class: str | None = None,
    project_root: str | None = None,
    project_name: str | None = None,
) -> dict:
    cfg = AISettings(**cfg_dict)
    fc = folder_class if folder_class and folder_class != "unknown" else None
    if fc and not is_valid_species_folder(fc, project_name):
        fc = None
    use_all = settings.batch_use_all_models if settings.batch_accurate_mode else settings.auto_use_all_models
    try:
        anns, w, h, timing = run_pipeline(
            image_path,
            prompts=prompts,
            ai_settings=cfg,
            folder_class=fc,
            use_all_models=use_all,
            project_root=project_root,
            project_name=project_name,
        )
        return {
            "ok": True,
            "path": image_path,
            "annotations": [a.model_dump() for a in anns],
            "width": w,
            "height": h,
            "timing": timing,
        }
    except Exception as e:
        return {"ok": False, "path": image_path, "error": str(e)}


@dataclass
class BatchJob:
    job_id: str
    project_id: int
    # id, path, species_class, drive_file_id (optional)
    image_paths: list[tuple[int, str, str | None, str | None]]
    drive_access_token: str | None = None
    local_mirror_path: str | None = None
    status: str = "queued"
    completed: int = 0
    failed: int = 0
    total: int = 0
    errors: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    paused: bool = False
    cancelled: bool = False
    _run_active: bool = False
    start_time: float = 0
    current_image: str | None = None
    prompts: list[str] | None = None
    ai_settings: AISettings = field(default_factory=AISettings)


class BatchAnnotator:
    _jobs: dict[str, BatchJob] = {}

    def __init__(self, thread_pool_size: int = 4):
        self.pool_size = thread_pool_size

    def create_job(
        self,
        project_id: int,
        images: list[tuple[int, str, str | None, str | None]],
        prompts: list[str] | None = None,
        ai_settings: AISettings | None = None,
        drive_access_token: str | None = None,
        local_mirror_path: str | None = None,
    ) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = BatchJob(
            job_id=job_id,
            project_id=project_id,
            image_paths=images,
            total=len(images),
            prompts=prompts,
            ai_settings=ai_settings or AISettings(),
            drive_access_token=drive_access_token,
            local_mirror_path=local_mirror_path,
        )
        return job_id

    def _resolve_local_path(self, job: BatchJob, img_id: int, path: str, drive_file_id: str | None) -> str:
        from app.database import Project
        from app.media_resolver import resolve_image_path

        db = SessionLocal()
        try:
            proj = db.query(Project).filter(Project.id == job.project_id).first()
            img = db.query(ImageRecord).filter(ImageRecord.id == img_id).first()
            if not img:
                return path
            resolved, _ = resolve_image_path(
                proj,
                img,
                local_mirror=job.local_mirror_path,
                drive_token=job.drive_access_token,
            )
            return str(resolved) if resolved else path
        finally:
            db.close()

    def _sync_chunk_to_drive(self, job: BatchJob, chunk_meta: list[tuple[int, str]]) -> None:
        if not job.drive_access_token:
            return
        from app.database import Project
        from app.drive_sync import sync_image_annotations_to_drive

        db = SessionLocal()
        try:
            proj = db.query(Project).filter(Project.id == job.project_id).first()
            if not proj or getattr(proj, "source", "local") != "drive":
                return
            for img_id, _ in chunk_meta:
                img = db.query(ImageRecord).filter(ImageRecord.id == img_id).first()
                if img:
                    try:
                        sync_image_annotations_to_drive(db, proj, img, job.drive_access_token)
                    except Exception:
                        pass
        finally:
            db.close()

    def get_job(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def pause(self, job_id: str):
        job = self._jobs.get(job_id)
        if job:
            job.paused = True

    def resume(self, job_id: str):
        job = self._jobs.get(job_id)
        if job:
            job.paused = False

    def cancel(self, job_id: str):
        job = self._jobs.get(job_id)
        if job:
            job.cancelled = True
            job.status = "cancelled"

    def status(self, job_id: str) -> BatchStatus:
        job = self._jobs[job_id]
        elapsed = max(time.time() - job.start_time, 0.001) if job.start_time else 1
        done = min(job.completed + job.failed, job.total)
        completed = min(job.completed, job.total)
        ips = completed / elapsed if completed else 0
        remaining = max(0, job.total - done)
        eta = 0.0 if job.status in ("completed", "cancelled") else (remaining / ips if ips > 0 else 0)
        return BatchStatus(
            job_id=job_id,
            status=job.status,
            total=job.total,
            completed=completed,
            failed=job.failed,
            images_per_sec=round(ips, 2),
            eta_seconds=round(eta, 1),
            current_image=job.current_image,
            errors=job.errors[-50:],
            timing=job.timing,
        )

    def _save_result(self, image_id: int, result: dict):
        db = SessionLocal()
        try:
            img = db.query(ImageRecord).filter(ImageRecord.id == image_id).first()
            if not img:
                return
            db.query(Annotation).filter(Annotation.image_id == image_id).delete()
            for a in result.get("annotations", []):
                db.add(
                    Annotation(
                        image_id=image_id,
                        class_name=a.get("class_name", "unknown"),
                        confidence=a.get("confidence", 1),
                        x=a["x"],
                        y=a["y"],
                        w=a["w"],
                        h=a["h"],
                        rotation=a.get("rotation", 0),
                        source=a.get("source", "ai"),
                        attributes=a.get("attributes", {}),
                    )
                )
            img.status = "ai"
            img.width = result.get("width", img.width)
            img.height = result.get("height", img.height)
            db.commit()
        finally:
            db.close()

    def _save_chunk(self, chunk_meta: list[tuple[int, str]], results: list[dict]):
        """One DB transaction per chunk (much faster than per image)."""
        db = SessionLocal()
        try:
            for (img_id, path), result in zip(chunk_meta, results):
                if not result.get("ok"):
                    continue
                img = db.query(ImageRecord).filter(ImageRecord.id == img_id).first()
                if not img:
                    continue
                db.query(Annotation).filter(Annotation.image_id == img_id).delete()
                for a in result.get("annotations", []):
                    db.add(
                        Annotation(
                            image_id=img_id,
                            class_name=a.get("class_name", "unknown"),
                            confidence=a.get("confidence", 1),
                            x=a["x"],
                            y=a["y"],
                            w=a["w"],
                            h=a["h"],
                            rotation=a.get("rotation", 0),
                            source=a.get("source", "ai"),
                            attributes=a.get("attributes", {}),
                        )
                    )
                img.status = "ai"
                if result.get("width"):
                    img.width = result["width"]
                if result.get("height"):
                    img.height = result["height"]
            db.commit()
        finally:
            db.close()

    def _project_context(self, project_id: int) -> tuple[str | None, str | None]:
        from app.database import Project

        db = SessionLocal()
        try:
            proj = db.query(Project).filter(Project.id == project_id).first()
            if not proj:
                return None, None
            root = getattr(proj, "root_path", None) or getattr(proj, "path", None)
            return (str(root) if root else None), (proj.name if proj.name else None)
        finally:
            db.close()

    def _run_fast(self, job: BatchJob, on_progress: Callable | None):
        from app.batch_fast import annotate_batch_chunk

        cfg = job.ai_settings
        conf = max(cfg.confidence, settings.confidence, settings.min_detection_confidence)
        project_root, project_name = self._project_context(job.project_id)

        # One image at a time so progress updates (0/4 → 1/4 …) during slow CPU inference.
        for img_id, path, species, drive_id in job.image_paths:
            if job.cancelled:
                break
            while job.paused:
                time.sleep(0.2)

            resolved = self._resolve_local_path(job, img_id, path, drive_id)
            job.current_image = os.path.basename(resolved) or path
            if on_progress:
                on_progress(self.status(job.job_id))

            try:
                results = annotate_batch_chunk(
                    [(resolved, species)],
                    conf=conf,
                    max_det=cfg.max_boxes,
                    project_root=project_root,
                    project_name=project_name,
                )
                result = results[0] if results else {"ok": False, "path": path, "error": "no result"}
            except Exception as e:
                logger.exception("batch chunk failed for %s", path)
                result = {"ok": False, "path": path, "error": str(e)}

            if result.get("ok"):
                self._save_result(img_id, result)
                self._sync_chunk_to_drive(job, [(img_id, path)])
                job.completed += 1
            else:
                job.failed += 1
                job.errors.append({"path": path, "error": result.get("error", "unknown")})

            if on_progress:
                on_progress(self.status(job.job_id))

        job.status = "cancelled" if job.cancelled else "completed"

    def _run_legacy(self, job: BatchJob, on_progress: Callable | None):
        cfg_dict = job.ai_settings.model_dump()
        project_root, project_name = self._project_context(job.project_id)
        pool_size = min(job.ai_settings.thread_pool_size or self.pool_size, 4)
        with ThreadPoolExecutor(max_workers=pool_size) as ex:
            futures = {
                ex.submit(
                    _annotate_one,
                    self._resolve_local_path(job, img_id, path, drive_id),
                    job.prompts,
                    cfg_dict,
                    species,
                    project_root,
                    project_name,
                ): (img_id, path)
                for img_id, path, species, drive_id in job.image_paths
            }
            for fut in as_completed(futures):
                if job.cancelled:
                    break
                while job.paused:
                    time.sleep(0.2)
                img_id, path = futures[fut]
                job.current_image = path
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"ok": False, "path": path, "error": str(e)}

                if result.get("ok"):
                    self._save_result(img_id, result)
                    self._sync_chunk_to_drive(job, [(img_id, path)])
                    for k, v in result.get("timing", {}).items():
                        job.timing[k] = job.timing.get(k, 0) + v
                    job.completed += 1
                else:
                    job.failed += 1
                    job.errors.append({"path": path, "error": result.get("error", "unknown")})

                if on_progress:
                    on_progress(self.status(job.job_id))

        job.status = "cancelled" if job.cancelled else "completed"

    def _run_sync(self, job: BatchJob, on_progress: Callable | None):
        if settings.batch_use_legacy_pipeline:
            self._run_legacy(job, on_progress)
        elif settings.batch_fast_mode:
            self._run_fast(job, on_progress)
        else:
            self._run_legacy(job, on_progress)

    async def run_async(self, job_id: str, on_progress: Callable | None = None):
        job = self._jobs[job_id]
        if job.status in ("completed", "cancelled"):
            if on_progress:
                await on_progress(self.status(job_id))
            return
        if job._run_active:
            while job._run_active:
                await asyncio.sleep(0.25)
                if on_progress:
                    await on_progress(self.status(job_id))
            if on_progress:
                await on_progress(self.status(job_id))
            return

        job._run_active = True
        job.status = "running"
        job.start_time = time.time()
        loop = asyncio.get_event_loop()

        def sync_progress(status):
            if not on_progress:
                return
            fut = asyncio.run_coroutine_threadsafe(on_progress(status), loop)
            try:
                fut.result(timeout=5.0)
            except Exception:
                pass

        try:
            await loop.run_in_executor(
                None, lambda: self._run_sync(job, sync_progress if on_progress else None)
            )
        except Exception as exc:
            logger.exception("batch job %s crashed", job_id)
            job.errors.append({"error": str(exc)})
            if job.status == "running":
                job.status = "completed"
        finally:
            job._run_active = False
        if on_progress:
            try:
                await on_progress(self.status(job_id))
            except Exception:
                pass


batch_annotator = BatchAnnotator()


async def run_batch_job(job_id: str) -> None:
    """Background entry point for POST /batch/start."""
    try:
        await batch_annotator.run_async(job_id)
    except Exception:
        logger.exception("run_batch_job failed for %s", job_id)
