#!/usr/bin/env python3
"""
Train a per-crop species classifier from a folder-organized dataset.

Layout:
  dataset_root/
    chromis_viridis/*.jpg
    datnioides/*.jpg
    ...

Usage:
  python scripts/train_crop_classifier.py "D:/path/to/dataset" --epochs 40

Output: backend/species_crop_cls.pt (and copy to ~/.marine-annotation-studio/models/)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_classification_dataset(source_root: Path, out_dir: Path, val_ratio: float = 0.15) -> int:
    """Ultralytics classify layout: train/<class>/img.jpg, val/<class>/img.jpg"""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    count = 0
    for species_dir in sorted(source_root.iterdir()):
        if not species_dir.is_dir():
            continue
        images = [p for p in species_dir.iterdir() if p.suffix.lower() in IMAGE_EXT]
        if not images:
            continue
        random.shuffle(images)
        n_val = max(1, int(len(images) * val_ratio)) if len(images) > 3 else 0
        val_set = set(images[:n_val])
        for img in images:
            split = val_dir if img in val_set else train_dir
            dest = split / species_dir.name / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train species crop classifier")
    parser.add_argument("dataset_root", type=Path, help="Root with one subfolder per species")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "species_crop_cls.pt")
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise SystemExit(f"Not a directory: {args.dataset_root}")

    work = Path(__file__).resolve().parent.parent / ".train_cls_cache"
    n = build_classification_dataset(args.dataset_root, work)
    if n < 50:
        raise SystemExit(f"Too few images ({n}). Need at least ~50 crops across species folders.")

    from ultralytics import YOLO

    print(f"Training on {n} images from {args.dataset_root} ...")
    model = YOLO(args.model)
    model.train(
        data=str(work),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(work / "runs"),
        name="species_cls",
        exist_ok=True,
        verbose=True,
    )

    best = work / "runs" / "species_cls" / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit("Training finished but best.pt not found.")

    shutil.copy2(best, args.out)
    models_home = Path.home() / ".marine-annotation-studio" / "models" / "species_crop_cls.pt"
    models_home.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, models_home)
    print(f"Saved classifier to:\n  {args.out}\n  {models_home}")


if __name__ == "__main__":
    main()
