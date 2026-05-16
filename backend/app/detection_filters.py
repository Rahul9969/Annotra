"""Filter false-positive boxes (coral, low confidence, huge regions)."""
from __future__ import annotations

from app.model_ensemble import EnsembleDetection

# Model text that should not become a fish bounding box
# Exact labels (or prefix) that are not fish — do not use "reef" (blocks "reef fish")
NON_FISH_LABELS = frozenset({
    "coral",
    "sponge",
    "sea urchin",
    "starfish",
    "sea star",
    "anemone",
    "jellyfish",
    "marine invertebrate",
    "mollusk",
    "sea cucumber",
    "algae",
    "seaweed",
})


def is_fish_like(det: EnsembleDetection) -> bool:
    raw = (det.detected_as or det.class_name or "").strip().lower()
    if not raw:
        return True
    if raw in NON_FISH_LABELS:
        return False
    if " fish" in raw or raw.endswith("fish"):
        return True
    return not any(raw == kw or raw.startswith(f"{kw} ") for kw in NON_FISH_LABELS)


def clamp_detection(det: EnsembleDetection, image_w: int, image_h: int) -> EnsembleDetection:
    """Clip box to image bounds."""
    if image_w < 1 or image_h < 1:
        return det
    x1 = max(0.0, min(float(det.x), float(image_w - 1)))
    y1 = max(0.0, min(float(det.y), float(image_h - 1)))
    x2 = max(x1 + 1.0, min(float(det.x + det.w), float(image_w)))
    y2 = max(y1 + 1.0, min(float(det.y + det.h), float(image_h)))
    return EnsembleDetection(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        class_name=det.class_name,
        confidence=det.confidence,
        source=det.source,
        detected_as=det.detected_as,
    )


def box_area_ratio(det: EnsembleDetection, image_w: int, image_h: int) -> float:
    return (det.w * det.h) / max(1, image_w * image_h)


def filter_detections(
    dets: list[EnsembleDetection],
    image_w: int,
    image_h: int,
    min_confidence: float,
    max_box_area_ratio: float = 0.38,
) -> list[EnsembleDetection]:
    if not dets:
        return []
    img_area = max(1, image_w * image_h)
    kept: list[EnsembleDetection] = []
    for d in dets:
        d = clamp_detection(d, image_w, image_h)
        if d.confidence < min_confidence:
            continue
        if not is_fish_like(d):
            continue
        box_area = d.w * d.h
        if box_area / img_area > max_box_area_ratio:
            continue
        if d.w < 4 or d.h < 4:
            continue
        kept.append(d)
    return kept


def pick_best_detection(
    dets: list[EnsembleDetection],
    image_w: int,
    image_h: int,
    *,
    max_box_area_ratio: float = 0.38,
    min_confidence: float = 0.12,
) -> EnsembleDetection | None:
    """Highest-confidence box that passes size/conf filters (last resort)."""
    candidates = filter_detections(
        dets, image_w, image_h, min_confidence, max_box_area_ratio
    )
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)
