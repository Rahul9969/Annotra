"""Interactive segmentation — magic wand (flood fill), SAM / GrabCut smart mask."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.model_paths import resolve_model_path

logger = logging.getLogger(__name__)

_sam_model = None
_sam_lock = threading.Lock()
_sam_status = "not_loaded"
_sam_error: str | None = None


@dataclass
class SegmentResult:
    polygon: list[list[float]]
    x: float
    y: float
    w: float
    h: float
    width: int
    height: int
    source: str


def sam_status() -> tuple[str, str | None]:
    return _sam_status, _sam_error


def _load_rgb(image_path: str) -> np.ndarray:
    with Image.open(image_path) as im:
        return np.array(im.convert("RGB"))


def _mask_to_polygon(mask: np.ndarray, simplify: float = 0.002) -> list[list[float]]:
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    eps = max(1.0, simplify * peri)
    approx = cv2.approxPolyDP(c, eps, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]


def _polygon_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    if not polygon:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return x1, y1, x2 - x1, y2 - y1


def _reject_oversized_mask(mask: np.ndarray, max_area_ratio: float = 0.88) -> np.ndarray:
    """Drop segments that cover nearly the whole frame (common flood-fill failure)."""
    import cv2

    h, w = mask.shape[:2]
    img_area = w * h
    if mask.sum() / 255 / img_area <= max_area_ratio:
        return mask
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Segment covers the full image — click directly on the fish body")
    c = max(contours, key=cv2.contourArea)
    tight = np.zeros_like(mask)
    cv2.drawContours(tight, [c], -1, 255, -1)
    if tight.sum() / 255 / img_area > max_area_ratio:
        raise ValueError(
            "Segment covers too much of the image — try Smart (click the fish) or lower magic tolerance"
        )
    return tight


def _result_from_mask(mask: np.ndarray, source: str) -> SegmentResult:
    h, w = mask.shape[:2]
    mask = _reject_oversized_mask(mask)
    polygon = _mask_to_polygon(mask)
    if not polygon:
        raise ValueError("Empty segmentation mask")
    if len(polygon) < 3:
        raise ValueError("Could not build a valid polygon from the segment")
    x, y, bw, bh = _polygon_bbox(polygon)
    return SegmentResult(
        polygon=polygon,
        x=x,
        y=y,
        w=bw,
        h=bh,
        width=w,
        height=h,
        source=source,
    )


def magic_wand(
    image_np: np.ndarray,
    x: float,
    y: float,
    *,
    tolerance: int = 35,
) -> SegmentResult:
    import cv2

    h, w = image_np.shape[:2]
    xi = int(max(0, min(w - 1, round(x))))
    yi = int(max(0, min(h - 1, round(y))))
    work = image_np.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    tol = int(max(1, min(120, tolerance)))
    lo = (tol, tol, tol)
    hi = (tol, tol, tol)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    cv2.floodFill(work, flood_mask, (xi, yi), (0, 0, 0), lo, hi, flags)
    mask = flood_mask[1:-1, 1:-1]
    if mask.sum() < 50:
        raise ValueError("Magic wand found no region — try another pixel or increase tolerance")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return _result_from_mask(mask, "magic")


def _grabcut_segment(
    image_np: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    h, w = image_np.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    fg_pts = [(int(round(p[0])), int(round(p[1]))) for p, lb in zip(points, labels) if lb == 1]
    if not fg_pts:
        raise ValueError("Add at least one foreground click (normal click)")

    xs = [max(0, min(w - 1, p[0])) for p in fg_pts]
    ys = [max(0, min(h - 1, p[1])) for p in fg_pts]
    margin = max(12, int(min(w, h) * 0.04))
    rx = max(0, min(xs) - margin)
    ry = max(0, min(ys) - margin)
    rw = min(w - rx, max(xs) - min(xs) + 2 * margin)
    rh = min(h - ry, max(ys) - min(ys) + 2 * margin)
    rw = max(4, rw)
    rh = max(4, rh)

    cv2.grabCut(bgr, mask, (rx, ry, rw, rh), bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    r = max(6, margin // 2)
    for (px, py) in fg_pts:
        cv2.circle(mask, (px, py), r, cv2.GC_FGD, -1)
    for p, lb in zip(points, labels):
        if lb != 0:
            continue
        px = max(0, min(w - 1, int(round(p[0]))))
        py = max(0, min(h - 1, int(round(p[1]))))
        cv2.circle(mask, (px, py), r, cv2.GC_BGD, -1)
    cv2.grabCut(bgr, mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _get_sam_model():
    global _sam_model, _sam_status, _sam_error
    if _sam_status == "ready" and _sam_model is not None:
        return _sam_model
    with _sam_lock:
        if _sam_model is not None:
            return _sam_model
        _sam_status = "loading"
        _sam_error = None
        try:
            from ultralytics import SAM

            path = resolve_model_path(settings.sam_model, min_bytes=1_000_000)
            if path:
                _sam_model = SAM(str(path))
                logger.info("SAM loaded from local weights: %s", path)
            else:
                corrupt = settings.models_dir / Path(settings.sam_model).name
                if corrupt.is_file() and corrupt.stat().st_size <= 1_000_000:
                    corrupt.unlink(missing_ok=True)
                _sam_model = SAM(settings.sam_model)
                logger.info("SAM loading by name (download if needed): %s", settings.sam_model)
            _sam_status = "ready"
        except Exception as e:
            _sam_status = "error"
            _sam_error = str(e)
            _sam_model = None
            logger.warning("SAM load failed, using GrabCut fallback: %s", e)
        return _sam_model


def _sam_segment(
    image_np: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
) -> np.ndarray | None:
    model = _get_sam_model()
    if model is None:
        return None
    try:
        from app.model_ensemble import ensemble

        device = ensemble.device
    except Exception:
        device = "cpu"

    pt_list = [[float(p[0]), float(p[1])] for p in points]
    lb_list = [int(l) for l in labels]
    results = model.predict(
        source=image_np,
        points=pt_list,
        labels=lb_list,
        verbose=False,
        device=device,
    )
    if not results or results[0].masks is None:
        return None
    masks = results[0].masks.data
    if masks is None or len(masks) == 0:
        return None
    m = masks[0].cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    return (m > 0.5).astype(np.uint8) * 255


def smart_segment(
    image_np: np.ndarray,
    points: list[tuple[float, float]],
    labels: list[int],
) -> SegmentResult:
    if not points or len(points) != len(labels):
        raise ValueError("points and labels are required")

    if settings.enable_sam_segment:
        try:
            mask = _sam_segment(image_np, points, labels)
            if mask is not None and mask.sum() > 50:
                return _result_from_mask(mask, "sam")
        except Exception as e:
            logger.debug("SAM segment failed: %s", e)

    mask = _grabcut_segment(image_np, points, labels)
    h, w = image_np.shape[:2]
    min_pixels = max(50, int(0.008 * w * h))
    if mask.sum() < min_pixels:
        fg = next((p for p, lb in zip(points, labels) if lb == 1), points[0])
        return magic_wand(
            image_np,
            fg[0],
            fg[1],
            tolerance=max(settings.magic_wand_tolerance, 28),
        )
    result = _result_from_mask(mask, "smart_cv")
    if result.w * result.h < 0.01 * w * h:
        fg = next((p for p, lb in zip(points, labels) if lb == 1), points[0])
        return magic_wand(
            image_np,
            fg[0],
            fg[1],
            tolerance=max(settings.magic_wand_tolerance, 28),
        )
    return result


def segment_from_path_magic(image_path: str, x: float, y: float, tolerance: int) -> SegmentResult:
    rgb = _load_rgb(image_path)
    return magic_wand(rgb, x, y, tolerance=tolerance)


def segment_from_path_smart(
    image_path: str,
    points: list[tuple[float, float]],
    labels: list[int],
) -> SegmentResult:
    rgb = _load_rgb(image_path)
    return smart_segment(rgb, points, labels)
