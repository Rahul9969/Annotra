"""Multi-model ensemble: custom PT, TFLite, YOLO-World open vocabulary."""
from __future__ import annotations

import logging
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings
from app.model_paths import resolve_model_path
from app.nms import nms_fusion
from app.species_catalog import get_catalog

logger = logging.getLogger(__name__)
_load_lock = threading.Lock()
_preload_thread: threading.Thread | None = None

_BACKEND = Path(__file__).resolve().parent.parent


@dataclass
class EnsembleDetection:
    x: float
    y: float
    w: float
    h: float
    class_name: str
    confidence: float
    source: str = "yolo"
    detected_as: str = ""  # raw class before folder relabel (for filtering coral etc.)
    polygon: list[list[float]] | None = None  # SAM2-derived mask polygon


def _valid_pt(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 500_000:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def _parse_model_list(raw: str) -> list[Path]:
    if not raw:
        return []
    out = []
    for part in raw.replace(";", ",").split(","):
        p = part.strip().strip('"')
        if not p:
            continue
        resolved = resolve_model_path(p, min_bytes=10_000)
        if resolved:
            out.append(resolved)
        else:
            path = Path(p).expanduser()
            if path.is_file():
                out.append(path.resolve())
    return out


def _load_vocab_classes() -> list[str]:
    return get_catalog().open_vocab_classes(None)


def _skip_world_in_fast_folder() -> bool:
    """Only skip World in fast folder mode when using the full 500+ species prompt list."""
    return (settings.open_vocab_prompt_mode or "generic").lower() == "full"


class ModelEnsemble:
    def __init__(self):
        self._models: list[tuple[str, Any, list[str]]] = []  # source, model, class_names
        self._device: str | None = None
        self.model_status = "not_loaded"
        self.model_error: str | None = None
        self.using_custom_weights = False
        self.loaded_sources: list[str] = []

    def resolve_device(self, pref: str = "auto") -> str:
        if pref and pref != "auto":
            return pref
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self.resolve_device(settings.device)
        return self._device

    def _register(self, source: str, path: Path, class_names: list[str] | None = None):
        from ultralytics import YOLO

        if path.suffix == ".pt" and not _valid_pt(path):
            logger.warning("Skip invalid PT: %s", path)
            return
        if not path.is_file():
            logger.warning("Model not found: %s", path)
            return
        logger.info("Loading [%s] %s", source, path)
        model = YOLO(str(path))
        names = class_names or (list(model.names.values()) if model.names else [])
        if source == "yolo_custom":
            self.using_custom_weights = True
        self._models.append((source, model, names))
        self.loaded_sources.append(source)

    def preload_async(self) -> None:
        """Load models in a background thread so /health can report loading state."""
        global _preload_thread
        if self.model_status == "ready":
            return
        with _load_lock:
            if _preload_thread is not None and _preload_thread.is_alive():
                return
            self.model_status = "loading"
            self.model_error = None

        def _run() -> None:
            try:
                self.preload()
            except Exception:
                pass

        _preload_thread = threading.Thread(target=_run, name="marine-model-preload", daemon=True)
        _preload_thread.start()

    def wait_until_ready(self, timeout: float | None = 600) -> bool:
        global _preload_thread
        if self.model_status == "ready":
            return True
        if _preload_thread is not None and _preload_thread.is_alive():
            _preload_thread.join(timeout=timeout)
        return self.model_status == "ready"

    def preload(self):
        with _load_lock:
            self._models.clear()
            self.loaded_sources.clear()
            self.using_custom_weights = False
            self.model_status = "loading"
            self.model_error = None
            try:
                if settings.custom_yolo:
                    custom = resolve_model_path(settings.custom_yolo, min_bytes=100_000)
                    if custom:
                        self._register("yolo_custom", custom)
                    else:
                        logger.warning("Custom YOLO not found: %s", settings.custom_yolo)

                for i, p in enumerate(_parse_model_list(settings.tflite_models)):
                    if p.suffix == ".tflite":
                        labels = _BACKEND / "labels_65.txt"
                        cn = labels.read_text(encoding="utf-8").strip().splitlines() if labels.is_file() else []
                        self._register(f"tflite_{i}", p, cn)

                for i, p in enumerate(_parse_model_list(settings.extra_pt_models)):
                    if p.suffix == ".pt":
                        self._register(f"extra_pt_{i}", p)

                if settings.enable_open_vocab:
                    ov_path = resolve_model_path(settings.open_vocab_model, min_bytes=1_000_000)
                    if not ov_path:
                        ov_path = settings.models_dir / Path(settings.open_vocab_model).name
                    vocab = _load_vocab_classes()
                    from ultralytics import YOLO

                    ov = YOLO(str(ov_path) if ov_path.is_file() else settings.open_vocab_model)
                    try:
                        ov.set_classes(vocab)
                    except Exception as e:
                        logger.warning("set_classes failed (%s), using default world classes", e)
                    self._models.append(("open_vocab", ov, vocab))
                    self.loaded_sources.append("open_vocab")
                    mode = settings.open_vocab_prompt_mode or "generic"
                    logger.info(
                        "Open-vocab model: %d prompts (mode=%s)",
                        len(vocab),
                        mode,
                    )

                elif settings.yolo_model:
                    base = Path(settings.yolo_model)
                    if not base.is_file():
                        base = settings.models_dir / base.name
                    self._register("yolo_base", base.resolve())

                if not self._models:
                    raise RuntimeError("No models loaded — check backend/.env paths")

                self.model_status = "ready"
                self.model_error = None
                logger.info("Ensemble ready: %s", self.loaded_sources)
            except Exception as e:
                self.model_status = "error"
                self.model_error = str(e)
                logger.exception("Ensemble preload failed")

    def _open_vocab_classes(self, folder_class: str | None) -> list[str]:
        return get_catalog().open_vocab_classes(folder_class)

    def predict_all(
        self,
        image_path: str,
        conf: float,
        max_det: int,
        folder_class: str | None = None,
        fast_folder: bool = False,
    ) -> tuple[list[EnsembleDetection], dict[str, float]]:
        if self.model_status == "loading":
            raise RuntimeError("AI models are still loading. Please wait and try again.")
        if self.model_status != "ready":
            raise RuntimeError(
                self.model_error or "AI models are not loaded. Check backend logs and .env paths."
            )

        import concurrent.futures

        timing: dict[str, float] = {}
        all_dets: list[EnsembleDetection] = []
        per_model = max_det

        per_model_nms = min(settings.yolo_nms_iou, 0.55)
        within_model_iou = 0.5

        def _run_yolo_model(source, model, class_names):
            t0 = time.perf_counter()
            model_timing = {}
            if fast_folder and source == "open_vocab" and _skip_world_in_fast_folder():
                model_timing["open_vocab_skipped"] = 1.0
                return [], model_timing
            if fast_folder and not settings.fast_mode_use_tflite and source.startswith("tflite"):
                model_timing[f"{source}_skipped"] = 1.0
                return [], model_timing

            try:
                if source == "open_vocab":
                    try:
                        ov_classes = self._open_vocab_classes(folder_class)
                        model.set_classes(ov_classes)
                        class_names = ov_classes
                    except Exception as e:
                        logger.warning("open_vocab set_classes: %s", e)

                if source == "open_vocab":
                    model_conf = max(conf, settings.open_vocab_min_confidence * 0.75)
                else:
                    model_conf = conf
                infer_imgsz = 640 if source.startswith("tflite") else settings.imgsz
                kwargs: dict[str, Any] = dict(
                    source=image_path,
                    conf=model_conf,
                    iou=per_model_nms,
                    max_det=per_model,
                    imgsz=infer_imgsz,
                    device=self.device,
                    half=settings.half_precision and "cuda" in self.device,
                    agnostic_nms=False,
                    verbose=False,
                )
                results = model.predict(**kwargs)
                model_dets: list[EnsembleDetection] = []
                for r in results:
                    if r.boxes is None:
                        continue
                    names = r.names or {}
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0])
                        score = float(box.conf[0])
                        if source == "open_vocab" and class_names:
                            raw_label = class_names[cls_id] if cls_id < len(class_names) else names.get(cls_id, "fish")
                        else:
                            raw_label = names.get(
                                cls_id,
                                class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                            )
                        raw_label = str(raw_label)
                        name = get_catalog().resolve_label(raw_label)
                        x1, y1, x2, y2 = xyxy
                        model_dets.append(
                            EnsembleDetection(
                                x=float(x1),
                                y=float(y1),
                                w=float(x2 - x1),
                                h=float(y2 - y1),
                                class_name=name,
                                confidence=score,
                                source=source,
                                detected_as=raw_label,
                            )
                        )
                model_dets = nms_fusion(model_dets, within_model_iou)
                model_timing[f"{source}_ms"] = (time.perf_counter() - t0) * 1000
                logger.debug("%s: %d boxes after per-model NMS", source, len(model_dets))
                return model_dets, model_timing
            except Exception as e:
                logger.warning("Model %s failed: %s", source, e)
                model_timing[f"{source}_error"] = 1.0
                return [], model_timing

        def _run_gdino():
            model_timing = {}
            if not settings.enable_grounding_dino:
                return [], model_timing
            try:
                from app.grounding_dino import detect as gdino_detect, build_prompt
                prompt = build_prompt(folder_class)
                gdino_dets, gdino_timing = gdino_detect(
                    image_path,
                    prompts=prompt,
                    box_threshold=settings.gdino_box_threshold,
                    text_threshold=settings.gdino_text_threshold,
                    folder_class=folder_class,
                )
                converted_dets = []
                for d in gdino_dets:
                    converted_dets.append(
                        EnsembleDetection(
                            x=d.x,
                            y=d.y,
                            w=d.w,
                            h=d.h,
                            class_name=d.class_name,
                            confidence=d.confidence,
                            source="grounding_dino",
                            detected_as=d.detected_as or d.class_name,
                        )
                    )
                model_timing.update(gdino_timing)
                return converted_dets, model_timing
            except Exception as e:
                logger.warning("Grounding DINO failed: %s", e)
                model_timing["grounding_dino_error"] = 1.0
                return [], model_timing

        # Run models concurrently to speed up CPU inference
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(self._models) + 1)) as executor:
            futures = []
            # Queue YOLO/TFLite models
            for source, model, class_names in self._models:
                futures.append(executor.submit(_run_yolo_model, source, model, class_names))
            
            # Queue Grounding DINO
            if settings.enable_grounding_dino:
                futures.append(executor.submit(_run_gdino))
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    dets, model_timing = future.result()
                    all_dets.extend(dets)
                    timing.update(model_timing)
                except Exception as e:
                    logger.warning("Ensemble thread failed: %s", e)

        return all_dets, timing

    def predict_open_vocab(
        self,
        image_path: str,
        conf: float,
        max_det: int,
        folder_class: str | None = None,
    ) -> tuple[list[EnsembleDetection], dict[str, float]]:
        """Run only YOLO-World (used when primary models find nothing)."""
        timing: dict[str, float] = {}
        for source, model, class_names in self._models:
            if source != "open_vocab":
                continue
            t0 = time.perf_counter()
            try:
                ov_classes = self._open_vocab_classes(folder_class)
                model.set_classes(ov_classes)
                model_conf = max(0.12, conf * 0.65, settings.open_vocab_min_confidence * 0.55)
                results = model.predict(
                    source=image_path,
                    conf=model_conf,
                    iou=min(settings.yolo_nms_iou, 0.55),
                    max_det=max_det,
                    imgsz=settings.imgsz,
                    device=self.device,
                    half=settings.half_precision and "cuda" in self.device,
                    verbose=False,
                )
                dets: list[EnsembleDetection] = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0])
                        score = float(box.conf[0])
                        raw = ov_classes[cls_id] if cls_id < len(ov_classes) else "fish"
                        raw = str(raw)
                        name = get_catalog().resolve_label(raw)
                        x1, y1, x2, y2 = xyxy
                        dets.append(
                            EnsembleDetection(
                                x=float(x1),
                                y=float(y1),
                                w=float(x2 - x1),
                                h=float(y2 - y1),
                                class_name=name,
                                confidence=score,
                                source=source,
                                detected_as=raw,
                            )
                        )
                dets = nms_fusion(dets, 0.5)
                timing["open_vocab_rescue_ms"] = (time.perf_counter() - t0) * 1000
                return dets, timing
            except Exception as e:
                logger.warning("open_vocab rescue failed: %s", e)
                timing["open_vocab_rescue_error"] = 1.0
        return [], timing

    def predict_custom_yolo(
        self,
        image_path: str,
        conf: float = 0.2,
        max_det: int = 20,
    ) -> tuple[list[EnsembleDetection], dict[str, float]]:
        """Run only custom best.pt (low conf rescue before CV fallback)."""
        timing: dict[str, float] = {}
        for source, model, class_names in self._models:
            if source != "yolo_custom":
                continue
            t0 = time.perf_counter()
            try:
                results = model.predict(
                    source=image_path,
                    conf=conf,
                    iou=min(settings.yolo_nms_iou, 0.55),
                    max_det=max_det,
                    imgsz=settings.imgsz,
                    device=self.device,
                    half=settings.half_precision and "cuda" in self.device,
                    verbose=False,
                )
                dets: list[EnsembleDetection] = []
                for r in results:
                    if r.boxes is None:
                        continue
                    names = r.names or {}
                    for box in r.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0])
                        score = float(box.conf[0])
                        raw_label = names.get(
                            cls_id,
                            class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                        )
                        raw_label = str(raw_label)
                        name = get_catalog().resolve_label(raw_label)
                        x1, y1, x2, y2 = xyxy
                        dets.append(
                            EnsembleDetection(
                                x=float(x1),
                                y=float(y1),
                                w=float(x2 - x1),
                                h=float(y2 - y1),
                                class_name=name,
                                confidence=score,
                                source="yolo_custom",
                                detected_as=raw_label,
                            )
                        )
                dets = nms_fusion(dets, 0.5)
                timing["yolo_custom_rescue_ms"] = (time.perf_counter() - t0) * 1000
                return dets, timing
            except Exception as e:
                logger.warning("custom yolo rescue failed: %s", e)
                timing["yolo_custom_rescue_error"] = 1.0
        return [], timing


ensemble = ModelEnsemble()
