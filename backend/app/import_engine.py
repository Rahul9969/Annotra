"""Import YOLO, COCO, VOC, CSV, and Annotra export folders into a new project."""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image
from sqlalchemy.orm import Session

from app.database import Annotation, ClassLabel, ImageRecord, Project, seed_project_classes
from app.paths import norm_path
from app.species import species_from_image_path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


@dataclass
class ParsedAnnotation:
    class_name: str
    x: float
    y: float
    w: float
    h: float
    polygon: list[list[float]] | None = None
    confidence: float = 1.0


@dataclass
class ParsedImage:
    path: Path
    rel_path: str
    width: int
    height: int
    annotations: list[ParsedAnnotation] = field(default_factory=list)
    species_class: str = "unknown"


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size
    except OSError:
        return 1, 1


def _resolve_dataset_root(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    if detect_dataset_format(path):
        return path
    for child in sorted(path.iterdir()):
        if child.is_dir() and detect_dataset_format(child):
            return child
    raise ValueError(
        "Could not detect dataset format. Choose a folder with YOLO (images/ + labels/), "
        "COCO (instances.json), VOC (voc/.../annotations), or annotations.csv."
    )


def detect_dataset_format(root: Path) -> str | None:
    root = root.resolve()
    if (root / "export_manifest.json").exists():
        try:
            manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
            fmt = str(manifest.get("format", "")).lower()
            if fmt.startswith("yolo") or fmt in ("yolo", "yolo_v5", "yolo_v8", "yolo_v11"):
                if (root / "images").is_dir():
                    return "yolo"
            if fmt == "coco" and (root / "instances.json").exists():
                return "coco"
            if fmt == "voc":
                return "voc"
            if fmt == "csv" and (root / "annotations.csv").exists():
                return "csv"
        except (json.JSONDecodeError, OSError):
            pass
    if (root / "instances.json").exists():
        return "coco"
    if (root / "annotations.csv").exists():
        return "csv"
    if (root / "data.yaml").exists() or (
        (root / "classes.txt").exists() and (root / "images").is_dir()
    ):
        return "yolo"
    for p in root.glob("voc/*/annotations"):
        if p.is_dir() and any(p.glob("*.xml")):
            return "voc"
    if (root / "images").is_dir() and (root / "labels").is_dir():
        return "yolo"
    if _folder_has_images(root):
        return "images_only"
    return None


def _folder_has_images(root: Path) -> bool:
    for p in [root, root / "images"]:
        if not p.is_dir():
            continue
        for img in p.rglob("*"):
            if img.is_file() and img.suffix.lower() in IMAGE_EXT:
                return True
    return False


def _parse_images_only(root: Path) -> tuple[list[ParsedImage], list[str]]:
    images: list[ParsedImage] = []
    seen: set[str] = set()
    for base in (root / "images", root):
        if not base.is_dir():
            continue
        for img_path in base.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXT:
                continue
            key = norm_path(str(img_path))
            if key in seen:
                continue
            seen.add(key)
            w, h = _read_image_size(img_path)
            try:
                rel = str(img_path.relative_to(root))
            except ValueError:
                rel = img_path.name
            sp = species_from_image_path(str(img_path), str(root))
            images.append(
                ParsedImage(
                    path=img_path,
                    rel_path=rel,
                    width=w,
                    height=h,
                    species_class=sp if sp != "unknown" else "unknown",
                )
            )
    classes = sorted(
        {im.species_class for im in images if im.species_class != "unknown"},
        key=str.lower,
    )
    return images, classes


def _load_yolo_class_names(root: Path) -> list[str]:
    classes_file = root / "classes.txt"
    if classes_file.is_file():
        names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if names:
            return names
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        try:
            data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, dict):
                return [names[k] for k in sorted(names, key=lambda x: int(x) if str(x).isdigit() else x)]
            if isinstance(names, list):
                return [str(n) for n in names]
        except (yaml.YAMLError, OSError):
            pass
    return []


def _yolo_label_path(img_path: Path, root: Path) -> Path | None:
    try:
        rel = img_path.relative_to(root)
    except ValueError:
        return None
    parts = list(rel.parts)
    if parts and parts[0] == "images" and len(parts) >= 2:
        parts[0] = "labels"
        candidate = root / Path(*parts).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    stem = img_path.stem
    for split in ("train", "val", "test", ""):
        if split:
            candidate = root / "labels" / split / f"{stem}.txt"
        else:
            candidate = root / "labels" / f"{stem}.txt"
        if candidate.is_file():
            return candidate
    return None


def _parse_yolo_label_line(line: str, class_names: list[str], w: int, h: int) -> ParsedAnnotation | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(float(parts[0]))
    except ValueError:
        return None
    class_name = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
    nums = [float(x) for x in parts[1:]]
    if len(nums) == 4:
        cx, cy, nw, nh = nums
        bw, bh = nw * w, nh * h
        x = cx * w - bw / 2
        y = cy * h - bh / 2
        return ParsedAnnotation(class_name=class_name, x=x, y=y, w=bw, h=bh)
    if len(nums) >= 6 and len(nums) % 2 == 0:
        xs = [nums[i] * w for i in range(0, len(nums), 2)]
        ys = [nums[i + 1] * h for i in range(0, len(nums), 2)]
        polygon = [[xs[i], ys[i]] for i in range(len(xs))]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return ParsedAnnotation(
            class_name=class_name,
            x=x1,
            y=y1,
            w=max(1.0, x2 - x1),
            h=max(1.0, y2 - y1),
            polygon=polygon,
        )
    return None


def _parse_yolo_dataset(root: Path) -> tuple[list[ParsedImage], list[str]]:
    class_names = _load_yolo_class_names(root)
    images: list[ParsedImage] = []
    img_root = root / "images"
    search_roots = [img_root] if img_root.is_dir() else [root]
    seen: set[str] = set()
    for base in search_roots:
        for img_path in base.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXT:
                continue
            key = norm_path(str(img_path))
            if key in seen:
                continue
            seen.add(key)
            w, h = _read_image_size(img_path)
            try:
                rel = str(img_path.relative_to(root))
            except ValueError:
                rel = img_path.name
            parsed = ParsedImage(path=img_path, rel_path=rel, width=w, height=h)
            lbl = _yolo_label_path(img_path, root)
            if lbl and lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    ann = _parse_yolo_label_line(line, class_names, w, h)
                    if ann:
                        parsed.annotations.append(ann)
            sp = species_from_image_path(str(img_path), str(root))
            if sp != "unknown":
                parsed.species_class = sp
            elif parsed.annotations and class_names:
                parsed.species_class = parsed.annotations[0].class_name
            else:
                parsed.species_class = "unknown"
            images.append(parsed)
    if not class_names:
        found: set[str] = set()
        for im in images:
            for a in im.annotations:
                found.add(a.class_name)
            if im.species_class != "unknown":
                found.add(im.species_class)
        class_names = sorted(found, key=str.lower)
    return images, class_names


def _parse_coco_dataset(root: Path) -> tuple[list[ParsedImage], list[str]]:
    data = json.loads((root / "instances.json").read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    class_names = [categories[k] for k in sorted(categories)]
    by_image: dict[int, list[ParsedAnnotation]] = {}
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            continue
        x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cat = categories.get(ann.get("category_id"), "unknown")
        by_image.setdefault(int(ann["image_id"]), []).append(
            ParsedAnnotation(class_name=cat, x=x, y=y, w=bw, h=bh)
        )
    images: list[ParsedImage] = []
    coco_img_dir = root / "coco_images"
    for info in data.get("images", []):
        file_name = info.get("file_name") or ""
        if not file_name:
            continue
        candidates = [
            coco_img_dir / file_name,
            root / file_name,
            root / "images" / file_name,
        ]
        img_path = next((p for p in candidates if p.is_file()), None)
        if not img_path:
            for hit in root.rglob(file_name):
                if hit.is_file():
                    img_path = hit
                    break
        if not img_path:
            continue
        w = int(info.get("width") or 0) or _read_image_size(img_path)[0]
        h = int(info.get("height") or 0) or _read_image_size(img_path)[1]
        anns = by_image.get(int(info["id"]), [])
        try:
            rel = str(img_path.relative_to(root))
        except ValueError:
            rel = img_path.name
        species = anns[0].class_name if anns else species_from_image_path(str(img_path), str(root))
        images.append(
            ParsedImage(
                path=img_path,
                rel_path=rel,
                width=w,
                height=h,
                annotations=anns,
                species_class=species if species != "unknown" else "unknown",
            )
        )
    return images, class_names


def _parse_voc_dataset(root: Path) -> tuple[list[ParsedImage], list[str]]:
    images: list[ParsedImage] = []
    class_names_set: set[str] = set()
    for ann_dir in sorted(root.glob("voc/*/annotations")):
        if not ann_dir.is_dir():
            continue
        split = ann_dir.parent.name
        img_dir = ann_dir.parent / "images"
        for xml_path in ann_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                continue
            root_el = tree.getroot()
            filename = (root_el.findtext("filename") or xml_path.stem).strip()
            img_path = img_dir / filename
            if not img_path.is_file():
                for hit in root.rglob(filename):
                    if hit.is_file() and hit.suffix.lower() in IMAGE_EXT:
                        img_path = hit
                        break
            if not img_path.is_file():
                continue
            size = root_el.find("size")
            w = int(size.findtext("width") or 0) if size is not None else 0
            h = int(size.findtext("height") or 0) if size is not None else 0
            if not w or not h:
                w, h = _read_image_size(img_path)
            anns: list[ParsedAnnotation] = []
            for obj in root_el.findall("object"):
                name = (obj.findtext("name") or "unknown").strip()
                class_names_set.add(name)
                box = obj.find("bndbox")
                if box is None:
                    continue
                xmin = float(box.findtext("xmin") or 0)
                ymin = float(box.findtext("ymin") or 0)
                xmax = float(box.findtext("xmax") or 0)
                ymax = float(box.findtext("ymax") or 0)
                anns.append(
                    ParsedAnnotation(
                        class_name=name,
                        x=xmin,
                        y=ymin,
                        w=max(1.0, xmax - xmin),
                        h=max(1.0, ymax - ymin),
                    )
                )
            try:
                rel = str(img_path.relative_to(root))
            except ValueError:
                rel = f"{split}/{filename}"
            species = anns[0].class_name if anns else "unknown"
            images.append(
                ParsedImage(
                    path=img_path,
                    rel_path=rel,
                    width=w,
                    height=h,
                    annotations=anns,
                    species_class=species,
                )
            )
    return images, sorted(class_names_set, key=str.lower)


def _parse_csv_dataset(root: Path) -> tuple[list[ParsedImage], list[str]]:
    csv_path = root / "annotations.csv"
    by_file: dict[str, list[ParsedAnnotation]] = {}
    class_names_set: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("filename") or row.get("file_name") or "").strip()
            if not fname:
                continue
            try:
                x1 = float(row.get("x1") or 0)
                y1 = float(row.get("y1") or 0)
                x2 = float(row.get("x2") or 0)
                y2 = float(row.get("y2") or 0)
            except ValueError:
                continue
            cls = (row.get("class") or row.get("class_name") or "unknown").strip()
            class_names_set.add(cls)
            conf = float(row.get("confidence") or 1.0)
            by_file.setdefault(fname, []).append(
                ParsedAnnotation(
                    class_name=cls,
                    x=x1,
                    y=y1,
                    w=max(1.0, x2 - x1),
                    h=max(1.0, y2 - y1),
                    confidence=conf,
                )
            )
    images: list[ParsedImage] = []
    for fname, anns in by_file.items():
        img_path = next((p for p in root.rglob(fname) if p.is_file()), None)
        if not img_path:
            continue
        w, h = _read_image_size(img_path)
        try:
            rel = str(img_path.relative_to(root))
        except ValueError:
            rel = fname
        images.append(
            ParsedImage(
                path=img_path,
                rel_path=rel,
                width=w,
                height=h,
                annotations=anns,
                species_class=anns[0].class_name if anns else "unknown",
            )
        )
    # Images listed only in folder without CSV rows — scan images/
    if (root / "images").is_dir():
        indexed = {norm_path(str(i.path)) for i in images}
        yolo_only, extra_classes = _parse_yolo_dataset(root)
        for im in yolo_only:
            if norm_path(str(im.path)) not in indexed:
                images.append(im)
                for a in im.annotations:
                    class_names_set.add(a.class_name)
    return images, sorted(class_names_set, key=str.lower)


def parse_dataset(root: Path) -> tuple[str, list[ParsedImage], list[str]]:
    fmt = detect_dataset_format(root)
    if not fmt:
        raise ValueError("Unsupported or unrecognized dataset layout.")
    if fmt == "yolo":
        images, classes = _parse_yolo_dataset(root)
    elif fmt == "coco":
        images, classes = _parse_coco_dataset(root)
    elif fmt == "voc":
        images, classes = _parse_voc_dataset(root)
    elif fmt == "csv":
        images, classes = _parse_csv_dataset(root)
    elif fmt == "images_only":
        images, classes = _parse_images_only(root)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if not images:
        raise ValueError("No images found in the selected folder.")
    return fmt, images, classes


def import_dataset(
    db: Session,
    dataset_path: str | Path,
    name: str | None = None,
) -> dict:
    """Create a project from an exported or standard YOLO/COCO/VOC/CSV dataset folder."""
    root = _resolve_dataset_root(Path(dataset_path))
    fmt, parsed_images, class_names = parse_dataset(root)
    project_name = (name or root.name).strip() or "Imported dataset"
    root_str = norm_path(str(root))

    existing = db.query(Project).filter(Project.root_path == root_str).first()
    if existing:
        proj = existing
        proj.name = project_name
        old_ids = [
            row.id
            for row in db.query(ImageRecord.id).filter(ImageRecord.project_id == proj.id).all()
        ]
        if old_ids:
            db.query(Annotation).filter(Annotation.image_id.in_(old_ids)).delete(
                synchronize_session=False
            )
        db.query(ImageRecord).filter(ImageRecord.project_id == proj.id).delete(
            synchronize_session=False
        )
        db.commit()
    else:
        proj = Project(name=project_name, root_path=root_str, source="local")
        db.add(proj)
        db.commit()
        db.refresh(proj)

    seed_project_classes(db, proj.id, class_names, folder_names_only=True, replace=True)
    class_rows = db.query(ClassLabel).filter(ClassLabel.project_id == proj.id).all()
    class_name_to_id = {c.name: c.id for c in class_rows}

    annotated_count = 0
    for item in parsed_images:
        status = "verified" if item.annotations else "unannotated"
        if item.annotations:
            annotated_count += 1
        img_row = ImageRecord(
            project_id=proj.id,
            path=norm_path(str(item.path)),
            rel_path=item.rel_path,
            width=item.width,
            height=item.height,
            status=status,
            species_class=item.species_class,
        )
        db.add(img_row)
        db.flush()
        for z, ann in enumerate(item.annotations):
            db.add(
                Annotation(
                    image_id=img_row.id,
                    class_id=class_name_to_id.get(ann.class_name, 0),
                    class_name=ann.class_name,
                    confidence=ann.confidence,
                    x=ann.x,
                    y=ann.y,
                    w=ann.w,
                    h=ann.h,
                    polygon=ann.polygon,
                    source="import",
                    z_index=z,
                )
            )
    db.commit()

    return {
        "project_id": proj.id,
        "name": proj.name,
        "root_path": proj.root_path,
        "format": fmt,
        "image_count": len(parsed_images),
        "annotated_count": annotated_count,
        "class_count": len(class_names),
    }
