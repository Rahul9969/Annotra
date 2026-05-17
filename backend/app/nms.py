"""Non-maximum suppression for multi-model fish detection."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class _HasBox(Protocol):
    x: float
    y: float
    w: float
    h: float
    confidence: float
    source: str


def _box_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    iou = inter / (union + 1e-6)
    ioa = inter / (min(area_a, area_b) + 1e-6)
    cx_a, cy_a = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    diag = np.sqrt(area_a) + np.sqrt(area_b)
    center_dist = float(np.hypot(cx_a - cx_b, cy_a - cy_b) / (diag + 1e-6))
    return float(iou), float(ioa), center_dist


def boxes_overlap(a: np.ndarray, b: np.ndarray, iou_thresh: float) -> bool:
    iou, ioa, center_dist = _box_metrics(a, b)
    if iou >= iou_thresh or ioa >= 0.72:
        return True
    if center_dist < 0.22 and iou >= 0.15:
        return True
    return False


def nms_fusion(detections: list[_HasBox], iou_thresh: float = 0.45) -> list:
    """Keep highest-confidence box when detections overlap (same fish from multiple models)."""
    if not detections:
        return []
    boxes = np.array([[d.x, d.y, d.x + d.w, d.y + d.h] for d in detections], dtype=np.float32)
    scores = np.array([d.confidence for d in detections], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        mask = []
        for j in rest:
            if boxes_overlap(boxes[i], boxes[j], iou_thresh):
                _, ioa, _ = _box_metrics(boxes[i], boxes[j])
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                # Nested partial box vs fuller detection — prefer larger when scores are close
                if ioa >= 0.55 and area_j > area_i * 1.12 and scores[j] >= scores[i] - 0.18:
                    i = j
                    keep[-1] = i
                mask.append(False)
                continue
            mask.append(True)
        order = rest[np.array(mask, dtype=bool)]
    merged = [detections[i] for i in keep]
    for d in merged:
        if getattr(d, "source", "") not in ("folder",):
            d.source = "ensemble"
    return merged
