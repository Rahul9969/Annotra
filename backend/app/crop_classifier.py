"""Per-crop species verifier — overrides folder label only when confident of a different species."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.model_ensemble import EnsembleDetection
from app.species_catalog import get_catalog

logger = logging.getLogger(__name__)

_classifier = None
_class_names: list[str] = []


def _load_classifier():
    global _classifier, _class_names
    if _classifier is not None:
        return _classifier
    path = settings.crop_classifier_weights
    if not path.is_file():
        path = settings.models_dir / "species_crop_cls.pt"
    if not path.is_file():
        logger.info("No crop classifier weights at %s — using folder labels only", path)
        return None
    try:
        from ultralytics import YOLO

        _classifier = YOLO(str(path))
        names = _classifier.names or {}
        _class_names = [names[i] for i in sorted(names.keys())] if names else []
        logger.info("Crop classifier loaded: %s (%d classes)", path, len(_class_names))
        return _classifier
    except Exception as e:
        logger.warning("Crop classifier load failed: %s", e)
        return None


def _crop_rgb(image_np: np.ndarray, det: EnsembleDetection, pad: float = 0.08) -> Image.Image | None:
    h, w = image_np.shape[:2]
    px = int(det.w * pad)
    py = int(det.h * pad)
    x1 = max(0, int(det.x) - px)
    y1 = max(0, int(det.y) - py)
    x2 = min(w, int(det.x + det.w) + px)
    y2 = min(h, int(det.y + det.h) + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return Image.fromarray(image_np[y1:y2, x1:x2])


def refine_with_crop_classifier(
    image_np: np.ndarray,
    dets: list[EnsembleDetection],
    folder_class: str,
    *,
    force: bool = False,
) -> list[EnsembleDetection]:
    """
    Default: every box keeps folder_class (typical multi-fish, same species).
    If a trained crop classifier strongly disagrees, use its label instead.
    """
    if not dets or (not settings.enable_crop_classifier and not force):
        return dets

    model = _load_classifier()
    if model is None:
        return dets

    catalog = get_catalog()
    folder_key = folder_class.strip().lower()
    thresh = settings.crop_verify_threshold
    out: list[EnsembleDetection] = []

    for det in dets:
        crop = _crop_rgb(image_np, det)
        if crop is None:
            det.class_name = folder_class
            out.append(det)
            continue
        try:
            results = model.predict(
                crop,
                verbose=False,
                device=settings.device if settings.device != "auto" else None,
            )
            if not results or results[0].probs is None:
                det.class_name = folder_class
                out.append(det)
                continue
            probs = results[0].probs
            top_i = int(probs.top1)
            top_conf = float(probs.top1conf)
            raw = _class_names[top_i] if top_i < len(_class_names) else str(top_i)
            predicted = catalog.resolve_label(raw)
            pred_key = predicted.strip().lower()

            if pred_key != folder_key and top_conf >= thresh:
                det.class_name = predicted
                det.confidence = top_conf
                det.source = "crop_cls"
            else:
                det.class_name = folder_class
            out.append(det)
        except Exception as e:
            logger.debug("Crop classify failed: %s", e)
            det.class_name = folder_class
            out.append(det)

    return out
