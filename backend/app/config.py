from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MARINE_",
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Annotra API"
    db_path: Path = Path.home() / ".marine-annotation-studio" / "annotations.db"
    models_dir: Path = Path.home() / ".marine-annotation-studio" / "models"
    confidence: float = 0.35
    min_detection_confidence: float = 0.32
    open_vocab_min_confidence: float = 0.38
    max_box_area_ratio: float = 0.38
    enable_detection_fallback: bool = False
    # Expand partial model boxes to full-fish CV union (tabletop / mesh backgrounds)
    enable_subject_box_expand: bool = False
    # When folder species is known but all models return 0 boxes, propose a box (CV fallback)
    folder_empty_fallback: bool = True
    iou_threshold: float = 0.55
    yolo_nms_iou: float = 0.55
    max_boxes: int = 300
    imgsz: int = 1280
    batch_size: int = 8
    thread_pool_size: int = 4
    device: str = "auto"
    half_precision: bool = True

    # Your trained 65-class model (fish market + some marine)
    custom_yolo: str | None = None

    # Comma-separated TFLite models (model_nano32.tflite, model32.tflite, ...)
    tflite_models: str = ""

    # Extra .pt models (comma-separated), optional
    extra_pt_models: str = ""

    # YOLO-World detector (yolov8s-worldv2.pt)
    enable_open_vocab: bool = True
    open_vocab_model: str = "yolov8s-worldv2.pt"
    # generic = ~25 marine prompts (fast, good boxes); full = 500+ species from CSV (slow)
    open_vocab_prompt_mode: str = "generic"
    open_vocab_classes_file: Path = _BACKEND_DIR / "data" / "marine_species_vocab.txt"
    species_mapping_csv: Path = _BACKEND_DIR / "data" / "species_mapping.csv"

    # Legacy single base model if open_vocab disabled
    yolo_model: str = "yolov8n.pt"
    dual_model: bool = True

    # folder_primary (default): all boxes = folder species; crop model may override rare mixed cases
    # smart | always | never
    folder_label_mode: str = "smart"
    # Skip YOLO-World when image has a known folder species (much faster)
    fast_folder_mode: bool = True
    # Single-image Auto: run custom PT + TFLite + YOLO-World (+ crop classifier if loaded)
    auto_use_all_models: bool = True
    # In fast mode, still run TFLite detectors (set false for fastest: custom PT only)
    fast_mode_use_tflite: bool = True
    # Per-crop classifier weights (train with scripts/train_crop_classifier.py)
    enable_crop_classifier: bool = True
    crop_classifier_weights: Path = _BACKEND_DIR / "species_crop_cls.pt"
    crop_verify_threshold: float = 0.72

    # Smart tool (SAM with GrabCut fallback) and magic wand
    enable_sam_segment: bool = True
    sam_model: str = "mobile_sam.pt"
    magic_wand_tolerance: int = 35

    # Grounding DINO — open-set text-prompted detection
    enable_grounding_dino: bool = False
    gdino_model: str = "IDEA-Research/grounding-dino-base"
    gdino_box_threshold: float = 0.30
    gdino_text_threshold: float = 0.25
    gdino_prompts: str = "fish . shrimp . crab . lobster . squid . octopus . marine creature"

    # SAM 2 — pixel-precise mask refinement for detected boxes
    enable_sam2: bool = False
    sam2_model: str = "sam2.1_b.pt"
    sam2_refine_boxes: bool = True
    annotation_mode: str = "both"  # bounding_box, segmentation, both

    # Batch: accurate mode = same fusion as single-image Auto (all models + rescue + box expand)
    batch_fast_mode: bool = True
    batch_accurate_mode: bool = True
    batch_use_all_models: bool = True
    batch_use_open_vocab: bool = True
    batch_imgsz: int = 960
    batch_accurate_imgsz: int = 1280
    batch_open_vocab_imgsz: int = 960
    batch_inference_size: int = 8
    # If true, batch runs full run_pipeline per image (slowest, highest parity with Auto)
    batch_use_legacy_pipeline: bool = True

    # Google Drive (OAuth + API). Create credentials at Google Cloud Console.
    google_client_id: str | None = None
    google_client_secret: str | None = None
    # Browser hits the backend first; must match an authorized redirect URI in the GCP client.
    google_oauth_redirect_uri: str = "http://127.0.0.1:8765/drive/oauth/callback"
    # After consent, user is redirected here with ?drive_oauth_state=<state> (Vite default port).
    google_oauth_frontend_redirect: str = "http://127.0.0.1:5173"

    # Cache Drive images locally for faster viewing and batch AI
    drive_cache_enabled: bool = True
    drive_cache_dir: Path = Path.home() / ".marine-annotation-studio" / "drive-cache"

    def model_post_init(self, __context: Any) -> None:
        from app.model_paths import resolve_model_path

        if self.custom_yolo:
            resolved = resolve_model_path(self.custom_yolo, min_bytes=100_000)
            if resolved:
                object.__setattr__(self, "custom_yolo", str(resolved))
        else:
            resolved = resolve_model_path("best.pt", min_bytes=100_000)
            if resolved:
                object.__setattr__(self, "custom_yolo", str(resolved))

        sam = resolve_model_path(self.sam_model, min_bytes=1_000_000)
        if sam:
            object.__setattr__(self, "sam_model", str(sam))

        ov = resolve_model_path(self.open_vocab_model, min_bytes=1_000_000)
        if ov:
            object.__setattr__(self, "open_vocab_model", str(ov))


settings = Settings()
settings.drive_cache_dir.mkdir(parents=True, exist_ok=True)
settings.db_path.parent.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
