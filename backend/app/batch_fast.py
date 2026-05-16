"""High-throughput batch annotation — batched multi-model inference + shared Auto post-process."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.model_ensemble import EnsembleDetection, ensemble
from app.pipeline import postprocess_detections
from app.species import is_valid_species_folder
from app.schemas import AISettings, AnnotationOut
from app.species_catalog import get_catalog

logger = logging.getLogger(__name__)
_inference_lock = threading.Lock()


def _model_entry(source: str) -> tuple[str, Any, list[str]] | None:
    for s, model, names in ensemble._models:
        if s == source:
            return s, model, names
    return None


def _tflite_sources() -> list[str]:
    return [s for s, _, _ in ensemble._models if s.startswith("tflite")]


def _boxes_from_result(result, source: str, class_names: list[str]) -> list[EnsembleDetection]:
    dets: list[EnsembleDetection] = []
    if result.boxes is None:
        return dets
    names = result.names or {}
    catalog = get_catalog()
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        score = float(box.conf[0])
        if source == "open_vocab" and class_names:
            raw = class_names[cls_id] if cls_id < len(class_names) else names.get(cls_id, "fish")
        else:
            raw = names.get(
                cls_id,
                class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
            )
        raw = str(raw)
        label = catalog.resolve_label(raw) if source == "open_vocab" else raw
        x1, y1, x2, y2 = xyxy
        dets.append(
            EnsembleDetection(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                class_name=label,
                confidence=score,
                source=source,
                detected_as=raw,
            )
        )
    return dets


def _batch_prompts(items: list[tuple[str, str | None]]) -> list[str]:
    """Shared World prompts for a chunk (generic fish + folder species in chunk)."""
    cat = get_catalog()
    merged: list[str] = []
    seen: set[str] = set()
    for _, fc in items:
        for p in cat.open_vocab_classes(fc if fc and fc != "unknown" else None):
            key = p.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(p.strip())
    return merged[:80]


def _predict_chunk(
    source: str,
    paths: list[str],
    conf: float,
    max_det: int,
    imgsz: int,
    infer_batch: int,
    ov_vocab: list[str] | None = None,
) -> list[Any]:
    entry = _model_entry(source)
    if not entry:
        return []
    src, model, class_names = entry
    if source == "open_vocab":
        if not ov_vocab:
            return []
        try:
            model.set_classes(ov_vocab)
            class_names = ov_vocab
        except Exception as e:
            logger.warning("batch set_classes: %s", e)
            return []
        conf = max(conf, settings.open_vocab_min_confidence)
    else:
        conf = max(conf, settings.min_detection_confidence)

    with _inference_lock:
        return model.predict(
            source=paths,
            conf=conf,
            iou=min(settings.yolo_nms_iou, 0.55),
            max_det=max_det,
            imgsz=imgsz,
            batch=infer_batch,
            device=ensemble.device,
            half=settings.half_precision and "cuda" in ensemble.device,
            agnostic_nms=False,
            verbose=False,
            stream=False,
        )


def annotate_batch_chunk(
    items: list[tuple[str, str | None]],
    conf: float | None = None,
    max_det: int | None = None,
    project_root: str | None = None,
    project_name: str | None = None,
) -> list[dict]:
    """
    Annotate many images per chunk: best.pt + TFLite + YOLO-World (when enabled),
    then the same postprocess path as single-image Auto.
    items: [(image_path, folder_class), ...]
    """
    if not items:
        return []

    if ensemble.model_status != "ready":
        raise RuntimeError(ensemble.model_error or "Models not ready")

    conf = conf if conf is not None else settings.confidence
    max_det = max_det or settings.max_boxes
    infer_batch = settings.batch_inference_size
    accurate = settings.batch_accurate_mode
    use_all = settings.batch_use_all_models and accurate
    custom_imgsz = settings.batch_accurate_imgsz if accurate else settings.batch_imgsz
    ai_cfg = AISettings(confidence=conf, max_boxes=max_det)

    paths = [str(Path(p).resolve()) for p, _ in items]
    folder_classes = [fc for _, fc in items]

    # Accurate batch: same per-image multi-model path as Auto (best boxes, no batched coord quirks).
    if accurate:
        out: list[dict] = []
        for path, folder_class in zip(paths, folder_classes):
            try:
                fc = folder_class if folder_class and folder_class != "unknown" else None
                if fc and not is_valid_species_folder(fc, project_name):
                    fc = None
                all_dets: list[EnsembleDetection] = []
                pred_timing: dict[str, float] = {}
                custom, t_custom = ensemble.predict_custom_yolo(
                    path,
                    conf=conf,
                    max_det=max_det,
                )
                pred_timing.update(t_custom)
                all_dets.extend(custom)
                if settings.batch_use_open_vocab and settings.enable_open_vocab:
                    ov, t_ov = ensemble.predict_open_vocab(
                        path, conf, max_det, fc
                    )
                    pred_timing.update(t_ov)
                    all_dets.extend(ov)
                with Image.open(path) as im:
                    image_np = np.array(im.convert("RGB"))
                height, width = image_np.shape[:2]
                fused, post_timing = postprocess_detections(
                    path,
                    image_np,
                    all_dets,
                    folder_class=folder_class,
                    project_root=project_root,
                    project_name=project_name,
                    use_all_models=use_all,
                    ai_settings=ai_cfg,
                )
                anns = [
                    AnnotationOut(
                        class_name=d.class_name,
                        confidence=d.confidence,
                        x=d.x,
                        y=d.y,
                        w=d.w,
                        h=d.h,
                        rotation=0,
                        source=d.source,
                    )
                    for d in fused
                ]
                timing = {"batch_per_image": 1.0, **pred_timing, **post_timing}
                out.append(
                    {
                        "ok": True,
                        "path": path,
                        "annotations": [a.model_dump() for a in anns],
                        "width": width,
                        "height": height,
                        "timing": timing,
                    }
                )
            except Exception as e:
                logger.exception("batch image failed: %s", path)
                out.append({"ok": False, "path": path, "error": str(e)})
        return out

    custom_entry = _model_entry("yolo_custom")
    if not custom_entry:
        raise RuntimeError("best.pt (MARINE_CUSTOM_YOLO) is not loaded")

    t0 = time.perf_counter()
    custom_results = _predict_chunk(
        "yolo_custom",
        paths,
        conf,
        max_det,
        custom_imgsz,
        infer_batch,
    )
    custom_ms = (time.perf_counter() - t0) * 1000

    tflite_ms: dict[str, float] = {}
    tflite_results: dict[str, list[Any]] = {}
    if use_all and settings.fast_mode_use_tflite:
        for src in _tflite_sources():
            t_tf = time.perf_counter()
            tflite_results[src] = _predict_chunk(
                src, paths, conf, max_det, 640, infer_batch
            )
            tflite_ms[src] = (time.perf_counter() - t_tf) * 1000

    world_results: list[Any] = []
    world_ms = 0.0
    vocab: list[str] = []
    use_world = settings.batch_use_open_vocab and settings.enable_open_vocab and _model_entry("open_vocab")
    if use_world:
        vocab = _batch_prompts(items)
        t1 = time.perf_counter()
        world_results = _predict_chunk(
            "open_vocab",
            paths,
            conf,
            max_det,
            settings.batch_open_vocab_imgsz,
            infer_batch,
            ov_vocab=vocab,
        )
        world_ms = (time.perf_counter() - t1) * 1000

    tf_summary = ", ".join(f"{k} {v:.0f}ms" for k, v in tflite_ms.items())
    logger.info(
        "Batch chunk %d imgs — custom %.0fms @%d%s%s",
        len(items),
        custom_ms,
        custom_imgsz,
        f", world {world_ms:.0f}ms" if use_world else "",
        f", {tf_summary}" if tf_summary else "",
    )

    out: list[dict] = []
    n = len(items)
    for i, (path, folder_class) in enumerate(zip(paths, folder_classes)):
        try:
            dets: list[EnsembleDetection] = []
            if i < len(custom_results):
                dets.extend(_boxes_from_result(custom_results[i], "yolo_custom", custom_entry[2]))
            for src, results in tflite_results.items():
                entry = _model_entry(src)
                if entry and i < len(results):
                    dets.extend(_boxes_from_result(results[i], src, entry[2]))
            if use_world and i < len(world_results):
                dets.extend(_boxes_from_result(world_results[i], "open_vocab", vocab))

            with Image.open(path) as im:
                image_np = np.array(im.convert("RGB"))
            height, width = image_np.shape[:2]

            fused, post_timing = postprocess_detections(
                path,
                image_np,
                dets,
                folder_class=folder_class,
                project_root=project_root,
                project_name=project_name,
                use_all_models=use_all,
                ai_settings=ai_cfg,
            )

            anns = [
                AnnotationOut(
                    class_name=d.class_name,
                    confidence=d.confidence,
                    x=d.x,
                    y=d.y,
                    w=d.w,
                    h=d.h,
                    rotation=0,
                    source=d.source,
                )
                for d in fused
            ]
            timing = {
                "batch_custom_ms": custom_ms / n,
                "batch_world_ms": world_ms / n if use_world else 0,
                "batch_accurate": 1.0 if accurate else 0.0,
                "batch_use_all_models": 1.0 if use_all else 0.0,
            }
            for src, ms in tflite_ms.items():
                timing[f"batch_{src}_ms"] = ms / n
            timing.update(post_timing)

            out.append(
                {
                    "ok": True,
                    "path": path,
                    "annotations": [a.model_dump() for a in anns],
                    "width": width,
                    "height": height,
                    "timing": timing,
                }
            )
        except Exception as e:
            logger.exception("batch image failed: %s", path)
            out.append({"ok": False, "path": path, "error": str(e)})

    return out
