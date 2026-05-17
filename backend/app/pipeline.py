"""Annotation pipeline — multi-model ensemble + folder labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.model_ensemble import EnsembleDetection, ensemble
from app.nms import _box_metrics, nms_fusion
from app.schemas import AnnotationOut, AISettings
from app.crop_classifier import refine_with_crop_classifier
from app.detection_filters import (
    clamp_detection,
    filter_detections,
    is_fish_like,
    pick_best_detection,
)
from app.species_catalog import get_catalog
from app.species import is_valid_species_folder, species_from_image_path


def _default_ai_settings() -> AISettings:
    return AISettings(
        confidence=settings.confidence,
        iou_threshold=settings.iou_threshold,
        max_boxes=settings.max_boxes,
        device=settings.device,
        half_precision=settings.half_precision,
        yolo_model=settings.yolo_model,
    )


def _texture_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Distinct blob on uniform sand/gravel (dark fish or object on bright substrate)."""
    import cv2

    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lap = np.abs(cv2.Laplacian(blur, cv2.CV_64F))
    lap = (lap / (lap.max() + 1e-6) * 255).astype(np.uint8)
    _, textured = cv2.threshold(lap, 25, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(dark, textured)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = w * h
    best = None
    best_score = 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.005 * img_area or area > 0.6 * img_area:
            continue
        if bw < 10 or bh < 10:
            continue
        score = area / img_area
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)
    return best


def _bright_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a bright subject (e.g. yellow fish on dark seabed)."""
    import cv2

    h, w = image_np.shape[:2]
    if h < 16 or w < 16:
        return None
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (12, 70, 90), (48, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = w * h
    best = None
    best_score = 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < 0.008 * img_area or area > 0.55 * img_area:
            continue
        if bw < 12 or bh < 12:
            continue
        score = area / img_area
        if score > best_score:
            best_score = score
            best = (x, y, bw, bh)
    return best


def _pad_xywh(x: int, y: int, bw: int, bh: int, w: int, h: int, pad_ratio: float = 0.04) -> tuple[int, int, int, int]:
    pad = max(6, int(min(bw, bh) * pad_ratio))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return x1, y1, x2 - x1, y2 - y1


def _union_contours_box(
    contours,
    img_area: int,
    w: int,
    h: int,
    *,
    min_area_ratio: float = 0.006,
    max_area_ratio: float = 0.88,
    min_side: int = 16,
) -> tuple[int, int, int, int] | None:
    import cv2

    x1, y1, x2, y2 = w, h, 0, 0
    found = False
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area_ratio * img_area or area > max_area_ratio * img_area:
            continue
        if bw < min_side or bh < min_side:
            continue
        found = True
        x1, y1 = min(x1, x), min(y1, y)
        x2, y2 = max(x2, x + bw), max(y2, y + bh)
    if not found or x2 <= x1 or y2 <= y1:
        return None
    return _pad_xywh(x1, y1, x2 - x1, y2 - y1, w, h)


def _union_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box around all sizable foreground blobs (whole fish vs small texture patch)."""
    import cv2

    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _union_contours_box(contours, w * h, w, h)


def _morph_union_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Merge split Otsu regions (head/tail) via morphological closing — better full-fish boxes."""
    import cv2

    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(9, int(min(w, h) * 0.018) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _union_contours_box(contours, w * h, w, h, min_area_ratio=0.008)


def _color_fish_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Red / orange / brown fish on neutral backgrounds (market & tabletop photos)."""
    import cv2

    h, w = image_np.shape[:2]
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    red_a = cv2.inRange(hsv, (0, 35, 45), (18, 255, 255))
    red_b = cv2.inRange(hsv, (165, 35, 45), (180, 255, 255))
    orange = cv2.inRange(hsv, (5, 28, 55), (28, 255, 230))
    mask = cv2.bitwise_or(cv2.bitwise_or(red_a, red_b), orange)
    k = max(9, int(min(w, h) * 0.015) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _union_contours_box(contours, w * h, w, h, min_area_ratio=0.012)


def _studio_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tabletop / mesh background: subject differs from border luminance and chroma."""
    import cv2

    h, w = image_np.shape[:2]
    if h < 32 or w < 32:
        return None
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    strip = max(4, int(min(h, w) * 0.02))
    border_l = np.concatenate(
        [L[:strip, :].ravel(), L[-strip:, :].ravel(), L[:, :strip].ravel(), L[:, -strip:].ravel()]
    )
    bg_l = float(np.median(border_l))
    chroma = np.sqrt((a.astype(np.float32) - 128) ** 2 + (b.astype(np.float32) - 128) ** 2)
    diff_l = np.abs(L.astype(np.float32) - bg_l)
    mask = ((diff_l > 10) | (chroma > 16)).astype(np.uint8) * 255
    k = max(11, int(min(w, h) * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _union_contours_box(contours, w * h, w, h, min_area_ratio=0.01)


def _best_subject_box(image_np: np.ndarray) -> tuple[int, int, int, int] | None:
    """Best single CV subject region (avoid union-of-all-heuristics megaboxes)."""
    h, w = image_np.shape[:2]
    img_area = w * h
    candidates: list[tuple[int, int, int, int]] = []
    for fn in (
        _morph_union_subject_box,
        _color_fish_subject_box,
        _studio_subject_box,
        _union_subject_box,
        _texture_subject_box,
        _bright_subject_box,
    ):
        box = fn(image_np)
        if box:
            candidates.append(box)
    if not candidates:
        return None
    # Prefer the largest plausible blob under ~42% of the frame.
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for x, y, bw, bh in candidates:
        area = bw * bh
        ratio = area / img_area
        if ratio < 0.008 or ratio > 0.42:
            continue
        scored.append((area, (x, y, bw, bh)))
    if not scored:
        return None
    _, best = max(scored, key=lambda t: t[0])
    x, y, bw, bh = best
    return _pad_xywh(x, y, bw, bh, w, h, pad_ratio=0.03)


def _expand_detection_to_subject(
    image_np: np.ndarray,
    det: EnsembleDetection,
) -> EnsembleDetection:
    """Grow partial YOLO/Open-Vocab boxes to cover the full fish silhouette when possible."""
    h, w = image_np.shape[:2]
    subj = _best_subject_box(image_np)
    if not subj:
        return det

    sx, sy, sw, sh = subj
    cx = det.x + det.w * 0.5
    cy = det.y + det.h * 0.5
    subj_area = sw * sh
    det_area = max(1.0, det.w * det.h)
    center_inside = sx <= cx <= sx + sw and sy <= cy <= sy + sh
    overlap = (
        max(0.0, min(det.x + det.w, sx + sw) - max(det.x, sx))
        * max(0.0, min(det.y + det.h, sy + sh) - max(det.y, sy))
    )
    iou = overlap / (det_area + subj_area - overlap + 1e-6)

    img_area = w * h
    subj_ratio = subj_area / max(1.0, img_area)
    if not center_inside and iou < 0.08:
        return det
    if subj_area < det_area * 1.06:
        return det
    if subj_ratio > min(settings.max_box_area_ratio, 0.42):
        return det
    if subj_area > det_area * 2.8:
        return det

    expanded = EnsembleDetection(
        x=float(sx),
        y=float(sy),
        w=float(sw),
        h=float(sh),
        class_name=det.class_name,
        confidence=det.confidence,
        source=det.source,
        detected_as=det.detected_as,
    )
    if det.confidence < 0.5 and det.source not in ("folder", "folder_cv"):
        expanded.source = "ensemble+subject"
    return expanded


def _expand_detections_to_subject(
    image_np: np.ndarray,
    dets: list[EnsembleDetection],
) -> list[EnsembleDetection]:
    if not dets:
        return dets
    return [_expand_detection_to_subject(image_np, d) for d in dets]


def _merge_partial_detections(
    dets: list[EnsembleDetection],
    *,
    image_w: int = 0,
    image_h: int = 0,
    iou_link: float = 0.06,
    ioa_link: float = 0.32,
    max_union_area_ratio: float = 0.36,
) -> list[EnsembleDetection]:
    """Union head/tail or multi-model fragments into one envelope before NMS picks a sliver."""
    if len(dets) < 2:
        return dets

    img_area = max(1, image_w * image_h)
    boxes = np.array([[d.x, d.y, d.x + d.w, d.y + d.h] for d in dets], dtype=np.float32)
    order = sorted(range(len(dets)), key=lambda i: dets[i].confidence, reverse=True)
    used = [False] * len(dets)
    merged: list[EnsembleDetection] = []

    for i in order:
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in order:
            if used[j] or j == i:
                continue
            iou, ioa, _ = _box_metrics(boxes[i], boxes[j])
            if iou >= iou_link or ioa >= ioa_link:
                cluster.append(j)
                used[j] = True
        if len(cluster) == 1:
            merged.append(dets[i])
            continue
        x1 = min(dets[k].x for k in cluster)
        y1 = min(dets[k].y for k in cluster)
        x2 = max(dets[k].x + dets[k].w for k in cluster)
        y2 = max(dets[k].y + dets[k].h for k in cluster)
        union_area = (x2 - x1) * (y2 - y1)
        if img_area > 1 and union_area / img_area > max_union_area_ratio:
            merged.append(dets[i])
            continue
        best = dets[i]
        merged.append(
            EnsembleDetection(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                class_name=best.class_name,
                confidence=max(dets[k].confidence for k in cluster),
                source="ensemble+merge",
                detected_as=best.detected_as,
            )
        )
    return merged


def _fallback_detection(image_np: np.ndarray, class_name: str) -> EnsembleDetection:
    import cv2

    h, w = image_np.shape[:2]
    union = _union_subject_box(image_np)
    if union:
        x, y, bw, bh = union
        return EnsembleDetection(
            x=float(x),
            y=float(y),
            w=float(bw),
            h=float(bh),
            class_name=class_name,
            confidence=0.58,
            source="folder_cv",
        )
    bright = _bright_subject_box(image_np) or _texture_subject_box(image_np)
    if bright:
        x, y, bw, bh = bright
        pad = max(4, int(min(bw, bh) * 0.08))
        return EnsembleDetection(
            x=float(max(0, x - pad)),
            y=float(max(0, y - pad)),
            w=float(min(w - x, bw + 2 * pad)),
            h=float(min(h - y, bh + 2 * pad)),
            class_name=class_name,
            confidence=0.52,
            source="folder_cv",
        )
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            if bw * bh >= 0.04 * w * h:
                pad = 4
                return EnsembleDetection(
                    x=float(max(0, x - pad)),
                    y=float(max(0, y - pad)),
                    w=float(min(w - x, bw + 2 * pad)),
                    h=float(min(h - y, bh + 2 * pad)),
                    class_name=class_name,
                    confidence=0.55,
                    source="folder",
                )
    except Exception:
        pass
    margin = 0.12
    mx, my = int(w * margin), int(h * margin)
    return EnsembleDetection(
        x=float(mx),
        y=float(my),
        w=float(w - 2 * mx),
        h=float(h - 2 * my),
        class_name=class_name,
        confidence=0.5,
        source="folder",
    )


def _use_fast_folder_path(folder_class: str | None) -> bool:
    return bool(
        settings.fast_folder_mode
        and folder_class
        and folder_class.strip().lower() not in ("", "unknown")
    )


def _apply_folder_label(
    dets: list[EnsembleDetection],
    folder_class: str | None,
    image_np: np.ndarray | None = None,
    *,
    run_crop_classifier: bool = False,
) -> list[EnsembleDetection]:
    if not folder_class or folder_class == "unknown" or not dets:
        return dets
    mode = (settings.folder_label_mode or "folder_primary").lower()

    if mode == "never":
        return dets

    if mode in ("always", "folder_primary"):
        fish_dets = [d for d in dets if is_fish_like(d)]
        if not fish_dets and dets:
            fish_dets = dets
        # Keep model class names (e.g. Black Pomfret) when folder is not a real species folder.
        if is_valid_species_folder(folder_class):
            for d in fish_dets:
                d.class_name = folder_class
        if mode == "folder_primary" and image_np is not None and (
            settings.enable_crop_classifier or run_crop_classifier
        ):
            return refine_with_crop_classifier(
                image_np, fish_dets, folder_class, force=run_crop_classifier
            )
        return fish_dets

    if mode == "smart":
        distinct = {d.class_name.strip().lower() for d in dets if d.class_name}
        if len(distinct) > 1:
            return dets
        if is_valid_species_folder(folder_class):
            for d in dets:
                d.class_name = folder_class
        return dets

    if is_valid_species_folder(folder_class):
        for d in dets:
            d.class_name = folder_class
    return dets


# Backward-compatible registry alias for routers
class _RegistryProxy:
    @property
    def device(self):
        return ensemble.device

    @property
    def model_status(self):
        return ensemble.model_status

    @property
    def model_error(self):
        return ensemble.model_error

    @property
    def using_custom_weights(self):
        return ensemble.using_custom_weights

    @property
    def dual_model_enabled(self):
        return len(ensemble.loaded_sources) > 1

    @property
    def _custom_path(self):
        return Path(settings.custom_yolo) if settings.custom_yolo else None

    @property
    def _base_path(self):
        return Path(settings.open_vocab_model) if settings.enable_open_vocab else Path(settings.yolo_model)

    @property
    def _custom_class_names(self):
        return []

    def preload_yolo(self):
        ensemble.preload()

    def hot_swap_yolo(self, model_path: str):
        settings.custom_yolo = model_path
        ensemble.preload_async()


registry = _RegistryProxy()


def _normalize_folder_class(
    folder_class: str | None,
    *,
    project_name: str | None,
    image_path: str,
    project_root: str | None,
) -> str | None:
    if folder_class and is_valid_species_folder(folder_class, project_name):
        return folder_class.strip()
    inferred = species_from_image_path(image_path, project_root)
    if is_valid_species_folder(inferred, project_name):
        return inferred
    return None


def postprocess_detections(
    image_path: str,
    image_np: np.ndarray,
    all_dets: list[EnsembleDetection],
    *,
    folder_class: str | None = None,
    project_root: str | None = None,
    project_name: str | None = None,
    use_all_models: bool = True,
    ai_settings: AISettings | None = None,
) -> tuple[list[EnsembleDetection], dict[str, float]]:
    """Shared fusion path for Auto and batch — NMS, rescue models, folder labels, CV box expand."""
    cfg = ai_settings or _default_ai_settings()
    height, width = image_np.shape[:2]
    timing: dict[str, float] = {}

    folder_class = _normalize_folder_class(
        folder_class,
        project_name=project_name,
        image_path=image_path,
        project_root=project_root,
    )
    predict_conf = max(cfg.confidence, settings.confidence)
    max_det = max(cfg.max_boxes, settings.max_boxes)
    fusion_iou = min(cfg.iou_threshold, settings.iou_threshold, 0.45)

    max_area = settings.max_box_area_ratio
    hard_max_area = 0.68
    dets = [clamp_detection(d, width, height) for d in all_dets]

    def _fuse_primary(detections: list[EnsembleDetection]) -> list[EnsembleDetection]:
        if len(detections) < 2:
            return list(detections)
        merged = _merge_partial_detections(
            detections, image_w=width, image_h=height, max_union_area_ratio=min(max_area, 0.38)
        )
        return nms_fusion(merged, fusion_iou)[:max_det]

    custom = [d for d in dets if d.source == "yolo_custom"]
    other = [d for d in dets if d.source != "yolo_custom"]
    custom_fused = filter_detections(
        _fuse_primary(custom),
        width,
        height,
        min_confidence=settings.min_detection_confidence,
        max_box_area_ratio=max_area,
    )
    if custom_fused:
        fused = list(custom_fused)
        for d in filter_detections(
            _fuse_primary(other),
            width,
            height,
            min_confidence=max(0.35, settings.min_detection_confidence),
            max_box_area_ratio=max_area,
        ):
            if d.confidence < 0.5:
                continue
            dup = False
            for c in fused:
                iou, ioa, _ = _box_metrics(
                    np.array([c.x, c.y, c.x + c.w, c.y + c.h]),
                    np.array([d.x, d.y, d.x + d.w, d.y + d.h]),
                )
                if iou > 0.35 or ioa > 0.5:
                    dup = True
                    break
            if not dup:
                fused.append(d)
        fused = fused[:max_det]
    else:
        if len(dets) > 1:
            dets = _merge_partial_detections(
                dets, image_w=width, image_h=height, max_union_area_ratio=min(max_area, 0.38)
            )
        fused = nms_fusion(dets, fusion_iou)[:max_det]
        fused = filter_detections(
            fused,
            width,
            height,
            min_confidence=settings.min_detection_confidence,
            max_box_area_ratio=max_area,
        )
    model_candidates = list(fused)

    if not fused and settings.enable_open_vocab:
        rescue, t_rescue = ensemble.predict_open_vocab(
            image_path, predict_conf, max_det, folder_class
        )
        timing.update(t_rescue)
        if rescue:
            fused = nms_fusion(rescue, fusion_iou)[:max_det]
            fused = filter_detections(
                fused,
                width,
                height,
                min_confidence=max(0.15, settings.min_detection_confidence * 0.75),
                max_box_area_ratio=max_area,
            )
            model_candidates = list(fused) or model_candidates

    if not fused and settings.custom_yolo and ensemble.using_custom_weights:
        rescue, t_custom = ensemble.predict_custom_yolo(
            image_path,
            conf=max(0.18, settings.min_detection_confidence * 0.45),
            max_det=max_det,
        )
        timing.update(t_custom)
        if rescue:
            fused = nms_fusion(rescue, fusion_iou)[:max_det]
            fused = filter_detections(
                fused,
                width,
                height,
                min_confidence=max(0.15, settings.min_detection_confidence * 0.5),
                max_box_area_ratio=max_area,
            )
            model_candidates = list(fused) or model_candidates

    fast_path = _use_fast_folder_path(folder_class) and not use_all_models
    if not fast_path:
        catalog = get_catalog()
        for d in fused:
            d.class_name = catalog.resolve_label(d.class_name)

    fused = _apply_folder_label(
        fused, folder_class, image_np, run_crop_classifier=use_all_models
    )

    use_fallback = settings.enable_detection_fallback or settings.folder_empty_fallback
    if not fused and use_fallback:
        label = folder_class if folder_class else "fish"
        try:
            fused = [_fallback_detection(image_np, label)]
            timing["fallback_ms"] = 1.0
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("fallback_detection failed: %s", exc)
            h, w = image_np.shape[:2]
            margin = 0.12
            mx, my = int(w * margin), int(h * margin)
            fused = [
                EnsembleDetection(
                    x=float(mx),
                    y=float(my),
                    w=float(w - 2 * mx),
                    h=float(h - 2 * my),
                    class_name=label,
                    confidence=0.45,
                    source="folder",
                )
            ]
            timing["fallback_margin_ms"] = 1.0

    pre_expand = list(fused)
    if fused and settings.enable_subject_box_expand:
        fused = _expand_detections_to_subject(image_np, fused)
        fused = nms_fusion(fused, min(fusion_iou, 0.4))[:max_det]
        timing["subject_expand_ms"] = 1.0

    # --- SAM2 mask refinement: tighten boxes + generate polygons ---
    if fused and settings.enable_sam2 and settings.sam2_refine_boxes and cfg.enable_sam:
        import time as _time

        _t0 = _time.perf_counter()
        try:
            from app.sam2_refiner import refine_detections as sam2_refine

            fused = sam2_refine(image_np, fused)
            timing["sam2_refine_ms"] = (_time.perf_counter() - _t0) * 1000
        except Exception as exc:
            import logging as _log

            _log.getLogger(__name__).warning("SAM2 refinement failed: %s", exc)
            timing["sam2_refine_error"] = 1.0

        # Run NMS again! Grounding DINO (loose box) and YOLO (tight box) might have had < 0.55 IoU
        # originally, but after SAM2 they will both shrink to the EXACT same tight box.
        # This second pass removes the newly overlapping duplicate.
        fused = nms_fusion(fused, 0.70)  # slightly higher IoU since SAM2 boxes should be nearly identical

    min_conf = predict_conf
    fused = filter_detections(fused, width, height, min_conf, hard_max_area)
    if not fused and pre_expand:
        fused = filter_detections(pre_expand, width, height, min_conf, hard_max_area)
    if not fused and model_candidates:
        best = pick_best_detection(
            model_candidates, width, height, max_box_area_ratio=hard_max_area, min_confidence=min_conf
        )
        if best:
            fused = [best]
            timing["best_model_box_ms"] = 1.0
    if not fused and all_dets:
        best = pick_best_detection(
            all_dets, width, height, max_box_area_ratio=hard_max_area, min_confidence=min_conf
        )
        if best:
            fused = [best]
            timing["best_raw_box_ms"] = 1.0

    return fused, timing


def run_pipeline(
    image_path: str,
    prompts: list[str] | None = None,
    ai_settings: AISettings | None = None,
    class_candidates: list[str] | None = None,
    folder_class: str | None = None,
    use_all_models: bool | None = None,
    project_root: str | None = None,
    project_name: str | None = None,
) -> tuple[list[AnnotationOut], int, int, dict[str, float]]:
    cfg = ai_settings or _default_ai_settings()
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        image_np = np.array(im)

    folder_class = _normalize_folder_class(
        folder_class,
        project_name=project_name,
        image_path=str(path),
        project_root=project_root,
    )

    predict_conf = max(cfg.confidence, settings.confidence)
    max_det = max(cfg.max_boxes, settings.max_boxes)

    if ensemble.model_status == "loading":
        raise RuntimeError("AI models are still loading. Please wait and try again.")
    if ensemble.model_status != "ready":
        raise RuntimeError(
            ensemble.model_error or "AI models are not loaded. Check backend/.env and restart."
        )

    use_all = settings.auto_use_all_models if use_all_models is None else use_all_models
    fast_path = _use_fast_folder_path(folder_class) and not use_all
    all_dets, timing = ensemble.predict_all(
        str(path),
        conf=predict_conf,
        max_det=max_det,
        folder_class=folder_class,
        fast_folder=fast_path,
    )
    timing["fast_folder_path"] = 1.0 if fast_path else 0.0
    timing["use_all_models"] = 1.0 if use_all else 0.0
    fused, post_timing = postprocess_detections(
        str(path),
        image_np,
        all_dets,
        folder_class=folder_class,
        project_root=project_root,
        project_name=project_name,
        use_all_models=use_all,
        ai_settings=cfg,
    )
    timing.update(post_timing)

    annotations = [
        AnnotationOut(
            class_name=d.class_name,
            confidence=d.confidence,
            x=d.x, y=d.y, w=d.w, h=d.h,
            rotation=0,
            source=d.source,
            polygon=getattr(d, "polygon", None),
        )
        for d in fused
    ]
    return annotations, width, height, timing
