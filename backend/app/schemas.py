from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    rotation: float = 0


class AnnotationOut(BaseModel):
    id: int | None = None
    class_id: int = 0
    class_name: str = "unknown"
    confidence: float = 1.0
    x: float
    y: float
    w: float
    h: float
    rotation: float = 0
    polygon: list[list[float]] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    source: str = "human"
    z_index: int = 0
    locked: bool = False
    hidden: bool = False


class ImageOut(BaseModel):
    id: int
    path: str
    rel_path: str | None = None
    drive_file_id: str | None = None
    width: int = 0
    height: int = 0
    status: str = "unannotated"
    annotation_count: int = 0


class ProjectCreate(BaseModel):
    name: str
    root_path: str
    class_names: list[str] | None = None


class ProjectOut(BaseModel):
    id: int
    name: str
    root_path: str
    image_count: int = 0
    annotated_count: int = 0
    source: str = "local"
    drive_folder_id: str | None = None
    local_mirror_path: str | None = None


class LocalMirrorBody(BaseModel):
    path: str | None = None


class RenameClassBody(BaseModel):
    old_name: str
    new_name: str


class ProjectDriveCreate(BaseModel):
    name: str
    folder_url: str
    access_token: str


class DriveOAuthRefresh(BaseModel):
    refresh_token: str


class AnnotateRequest(BaseModel):
    image_path: str
    image_id: int | None = None
    prompts: list[str] | None = None
    confidence: float | None = None
    project_id: int | None = None
    # When files live on collaborator PC only — raw base64 image bytes (no data: prefix)
    image_base64: str | None = None


class ShareTokenResponse(BaseModel):
    share_token: str


class ProjectShareLookup(BaseModel):
    project: ProjectOut
    share_token: str


class AnnotateResponse(BaseModel):
    annotations: list[AnnotationOut]
    width: int
    height: int
    timing_ms: dict[str, float] = Field(default_factory=dict)


class SegmentImageBody(BaseModel):
    image_path: str
    image_id: int | None = None
    project_id: int | None = None
    image_base64: str | None = None


class MagicSegmentRequest(SegmentImageBody):
    x: float
    y: float
    tolerance: int | None = None


class SmartSegmentRequest(SegmentImageBody):
    points: list[list[float]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)


class SegmentResponse(BaseModel):
    polygon: list[list[float]]
    x: float
    y: float
    w: float
    h: float
    width: int
    height: int
    source: str = "segment"


class BatchStartRequest(BaseModel):
    project_id: int
    image_ids: list[int] | None = None
    skip_annotated: bool = True
    thread_pool_size: int | None = None
    prompts: list[str] | None = None


class BatchStatus(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    failed: int
    images_per_sec: float = 0
    eta_seconds: float = 0
    current_image: str | None = None
    errors: list[dict[str, str]] = Field(default_factory=list)
    timing: dict[str, float] = Field(default_factory=dict)


class SaveAnnotationsRequest(BaseModel):
    image_id: int
    annotations: list[AnnotationOut]
    status: str | None = None


class AISettings(BaseModel):
    confidence: float = 0.1
    iou_threshold: float = 0.45
    max_boxes: int = 300
    batch_size: int = 8
    thread_pool_size: int = 4
    device: str = "auto"
    half_precision: bool = True
    yolo_model: str = "yolov8n.pt"
    custom_yolo: str | None = None
    enable_dino: bool = False
    enable_sam: bool = False
    enable_clip: bool = False


class ExportRequest(BaseModel):
    project_id: int
    output_dir: str
    format: str = "yolo_v8"
    split_train: float = 0.8
    split_val: float = 0.1
    confidence_min: float = 0.0
    reviewed_only: bool = False
    class_filter: list[str] | None = None
    include_labeled_previews: bool = False
    local_mirror_path: str | None = None
    create_zip: bool = True


class TrainRequest(BaseModel):
    project_id: int
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    model: str = "yolov8n.pt"


class StatsOut(BaseModel):
    total_images: int
    annotated_pct: float
    avg_boxes: float
    species_distribution: dict[str, int]
    status_counts: dict[str, int]
