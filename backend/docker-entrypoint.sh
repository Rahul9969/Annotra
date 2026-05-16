#!/bin/sh
set -e
cd /app

echo "[annotra] Checking model weights..."

if [ ! -f "best.pt" ]; then
  echo "ERROR: best.pt is missing in /app. Add it before docker build (see scripts/prepare-docker-build.ps1)."
  exit 1
fi

# Corrupt SAM checkpoints are a common Docker issue if the file was truncated during copy
if [ -f "mobile_sam.pt" ]; then
  sam_bytes=$(wc -c < mobile_sam.pt)
  if [ "$sam_bytes" -lt 1000000 ]; then
    echo "WARN: mobile_sam.pt looks too small (${sam_bytes} bytes) — Smart tool will use GrabCut fallback."
    echo "      Re-copy a valid mobile_sam.pt from Ultralytics and rebuild the image."
  fi
else
  echo "WARN: mobile_sam.pt missing — Smart segment tool will use GrabCut fallback."
fi

if [ ! -f "yolov8s-worldv2.pt" ]; then
  echo "[annotra] Downloading yolov8s-worldv2.pt (first run only)..."
  python -c "from ultralytics import YOLO; YOLO('yolov8s-worldv2.pt')"
fi

echo "[annotra] Weights OK — starting API."
exec "$@"
