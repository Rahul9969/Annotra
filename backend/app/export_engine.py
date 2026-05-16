"""Export annotations to YOLO, COCO, VOC, CSV, and annotated preview images."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from app.database import CLASS_COLORS, Annotation, ClassLabel, ImageRecord, Project
from app.media_resolver import MediaSource, resolve_image_path
from app.species import species_from_image_path

BOX_STROKE = "#FF2D95"
LABEL_BG = "#1a1a2e"

# (image row, annotations, folder class, resolved source file on disk)
ExportRow = tuple[ImageRecord, list[Annotation], str, Path]


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_") or "dataset"


def _folder_class_for_image(img: ImageRecord, project_root: str | None) -> str:
    sp = (img.species_class or "").strip()
    if not sp or sp.lower() == "unknown":
        sp = species_from_image_path(img.path, project_root)
    return sp if sp and sp.lower() != "unknown" else "unknown"


def _collect_subfolder_classes(db: Session, project_id: int, project_root: str | None) -> list[str]:
    images = db.query(ImageRecord).filter(ImageRecord.project_id == project_id).all()
    seen: set[str] = set()
    names: list[str] = []
    for img in images:
        sp = _folder_class_for_image(img, project_root)
        if sp != "unknown" and sp not in seen:
            seen.add(sp)
            names.append(sp)
    return sorted(names, key=lambda x: x.lower())


def _resolve_export_root(output_dir: str, project_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{_safe_name(project_name)}_export_{stamp}"


def _draw_preview(
    im: Image.Image,
    anns: list[Annotation],
    class_colors: dict[str, str],
    label_class: str | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for a in anns:
        label = label_class or a.class_name
        color = class_colors.get(label, BOX_STROKE)
        x1, y1 = int(a.x), int(a.y)
        x2, y2 = int(a.x + a.w), int(a.y + a.h)
        for w in (4, 2):
            draw.rectangle([x1, y1, x2, y2], outline=color, width=w)
        text_x = min(x2 + 6, im.width - 8)
        text_y = max(4, y1)
        bbox = draw.textbbox((text_x, text_y), label, font=font)
        pad = 2
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=LABEL_BG,
        )
        draw.text((text_x, text_y), label, fill=color, font=font)
    return im


def export_dataset(
    db: Session,
    project_id: int,
    output_dir: str,
    fmt: str = "yolo_v8",
    split_train: float = 0.8,
    split_val: float = 0.1,
    confidence_min: float = 0.0,
    reviewed_only: bool = False,
    class_filter: list[str] | None = None,
    include_labeled_previews: bool = False,
    *,
    local_mirror: str | None = None,
    drive_token: str | None = None,
    create_zip: bool = True,
) -> dict:
    proj = db.query(Project).filter(Project.id == project_id).first()
    project_name = proj.name if proj else "marine_dataset"
    out = _resolve_export_root(output_dir, project_name)
    out.mkdir(parents=True, exist_ok=True)

    project_root = proj.root_path if proj else None
    class_names = _collect_subfolder_classes(db, project_id, project_root)
    class_to_id: dict[str, int] = {n: i for i, n in enumerate(class_names)}

    db_classes = {
        c.name: c.color
        for c in db.query(ClassLabel).filter(ClassLabel.project_id == project_id).all()
    }
    class_colors = {
        name: db_classes.get(name, CLASS_COLORS[i % len(CLASS_COLORS)])
        for i, name in enumerate(class_names)
    }

    source_stats = {
        MediaSource.LOCAL_MIRROR.value: 0,
        MediaSource.LOCAL_PATH.value: 0,
        MediaSource.DRIVE_CACHE.value: 0,
        "skipped_no_file": 0,
        "skipped_no_annotations": 0,
    }

    images = db.query(ImageRecord).filter(ImageRecord.project_id == project_id).all()
    records: list[ExportRow] = []
    for img in images:
        if reviewed_only and img.status != "verified":
            continue
        folder_cls = _folder_class_for_image(img, project_root)
        if folder_cls == "unknown" or (class_filter and folder_cls not in class_filter):
            continue
        anns = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        filtered = [a for a in anns if a.confidence >= confidence_min]
        if not filtered or folder_cls not in class_to_id:
            if not filtered:
                source_stats["skipped_no_annotations"] += 1
            continue

        resolved, source = resolve_image_path(
            proj, img, local_mirror=local_mirror, drive_token=drive_token
        )
        if not resolved:
            source_stats["skipped_no_file"] += 1
            continue
        source_stats[source.value] += 1
        records.append((img, filtered, folder_cls, resolved))

    random.shuffle(records)
    n = len(records)
    n_train = int(n * split_train)
    n_val = int(n * split_val)
    splits: dict[str, list[ExportRow]] = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "format": fmt,
        "source_project": project_name,
        "note": "Annotated dataset export (images + labels). Original source folders were not modified.",
        "source_stats": source_stats,
        "local_mirror_used": bool(local_mirror),
        "files": [],
    }

    yolo_formats = ("yolo", "yolo_v8", "yolo_v11", "yolo_v5")
    if fmt in yolo_formats:
        _export_yolo(splits, out, class_names, class_to_id, manifest, fmt)
    elif fmt == "coco":
        _export_coco(splits, out, class_names, manifest)
    elif fmt == "voc":
        _export_voc(splits, out, class_names, manifest)
    elif fmt == "csv":
        _export_csv(splits, out, manifest)
    elif fmt == "labeled_images":
        _export_labeled(splits, out, manifest, class_colors)
    else:
        _export_yolo(splits, out, class_names, class_to_id, manifest, "yolo_v8")

    if include_labeled_previews and fmt != "labeled_images":
        _export_labeled(splits, out / "annotated_previews", manifest, class_colors, subfolder=False)

    readme = out / "README.txt"
    readme.write_text(
        f"Marine Annotation Studio export\n"
        f"Format: {fmt}\n"
        f"Annotated images: {n}\n"
        f"Image sources — local folder: {source_stats[MediaSource.LOCAL_MIRROR.value]}, "
        f"on-disk: {source_stats[MediaSource.LOCAL_PATH.value]}, "
        f"Drive cache: {source_stats[MediaSource.DRIVE_CACHE.value]}\n"
        f"Classes ({len(class_names)}): {', '.join(class_names[:20])}"
        f"{'...' if len(class_names) > 20 else ''}\n\n"
        f"YOLO: yolo detect train data=data.yaml model=yolo11n.pt\n",
        encoding="utf-8",
    )
    manifest_path = out / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path: str | None = None
    if create_zip and n > 0:
        zip_base = str(out)
        archive = shutil.make_archive(zip_base, "zip", root_dir=str(out))
        zip_path = archive

    return {
        "output_dir": str(out),
        "zip_path": zip_path,
        "manifest": str(manifest_path),
        "counts": {k: len(v) for k, v in splits.items()},
        "source_stats": source_stats,
        "total_exported": n,
    }


def _export_yolo(splits, out, class_names, class_to_id, manifest, fmt: str):
    (out / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")
    yolo_version = "v8" if "v8" in fmt else "v11" if "v11" in fmt else "v5" if "v5" in fmt else "v8"
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
        "nc": len(class_names),
        "roboflow": {"format": "yolo", "version": yolo_version},
    }
    header = f"# Ultralytics YOLO{yolo_version} object detection\n# Train: yolo detect train data=data.yaml model=yolo{yolo_version.replace('v','')}n.pt\n"
    (out / "data.yaml").write_text(header + yaml.dump(data_yaml, default_flow_style=False), encoding="utf-8")

    for split_name, items in splits.items():
        img_dir = out / "images" / split_name
        lbl_dir = out / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, anns, folder_cls, src in items:
            if not src.exists():
                continue
            dst = img_dir / src.name
            shutil.copy2(src, dst)
            w = img.width or 1
            h = img.height or 1
            if not img.width or not img.height:
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                except OSError:
                    pass
            lines = []
            cid = class_to_id.get(folder_cls, 0)
            for a in anns:
                cx = (a.x + a.w / 2) / w
                cy = (a.y + a.h / 2) / h
                nw, nh = a.w / w, a.h / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            manifest["files"].append({"split": split_name, "image": str(dst), "checksum": _checksum(dst)})


def _export_coco(splits, out, class_names, manifest):
    coco = {
        "info": {"description": "Marine Annotation Studio Export"},
        "licenses": [],
        "categories": [{"id": i + 1, "name": n, "supercategory": "marine"} for i, n in enumerate(class_names)],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    img_dir = out / "coco_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        for img, anns, folder_cls, src in items:
            if not src.exists():
                continue
            dst = img_dir / f"{split_name}_{src.name}"
            shutil.copy2(src, dst)
            image_id = len(coco["images"]) + 1
            w, h = img.width, img.height
            if not w or not h:
                try:
                    with Image.open(src) as im:
                        w, h = im.size
                except OSError:
                    w, h = 1, 1
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": dst.name,
                    "width": w,
                    "height": h,
                }
            )
            cat_id = class_names.index(folder_cls) + 1 if folder_cls in class_names else 1
            for a in anns:
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [a.x, a.y, a.w, a.h],
                        "area": a.w * a.h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
    (out / "instances.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    manifest["files"].append({"coco": str(out / "instances.json")})


def _export_voc(splits, out, class_names, manifest):
    for split_name, items in splits.items():
        voc_dir = out / "voc" / split_name
        img_dir = voc_dir / "images"
        ann_dir = voc_dir / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        for img, anns, folder_cls, src in items:
            if not src.exists():
                continue
            shutil.copy2(src, img_dir / src.name)
            stem = src.stem
            xml = ['<?xml version="1.0"?>', "<annotation>"]
            xml.append(f"<filename>{src.name}</filename>")
            w, h = img.width or 1, img.height or 1
            xml.append(f"<size><width>{w}</width><height>{h}</height><depth>3</depth></size>")
            for a in anns:
                xml.append("<object>")
                xml.append(f"<name>{folder_cls}</name>")
                xml.append(
                    f"<bndbox><xmin>{int(a.x)}</xmin><ymin>{int(a.y)}</ymin>"
                    f"<xmax>{int(a.x+a.w)}</xmax><ymax>{int(a.y+a.h)}</ymax></bndbox>"
                )
                xml.append("</object>")
            xml.append("</annotation>")
            (ann_dir / f"{stem}.xml").write_text("\n".join(xml), encoding="utf-8")


def _export_csv(splits, out, manifest):
    path = out / "annotations.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "rel_path", "x1", "y1", "x2", "y2", "class", "confidence", "split"])
        for split_name, items in splits.items():
            for img, anns, folder_cls, src in items:
                for a in anns:
                    w.writerow([
                        src.name,
                        img.rel_path or img.path,
                        a.x,
                        a.y,
                        a.x + a.w,
                        a.y + a.h,
                        folder_cls,
                        a.confidence,
                        split_name,
                    ])
    manifest["files"].append({"csv": str(path)})


def _export_labeled(splits, out, manifest, class_colors: dict[str, str], subfolder: bool = True):
    labeled_dir = out / "images" if subfolder else out
    labeled_dir.mkdir(parents=True, exist_ok=True)
    for split_name, items in splits.items():
        split_dir = labeled_dir / split_name if subfolder else labeled_dir
        split_dir.mkdir(parents=True, exist_ok=True)
        for img, anns, folder_cls, src in items:
            if not src.exists():
                continue
            im = Image.open(src).convert("RGB")
            _draw_preview(im, anns, class_colors, label_class=folder_cls)
            dst = split_dir / src.name
            im.save(dst, quality=92)
            manifest["files"].append({"labeled": str(dst)})
