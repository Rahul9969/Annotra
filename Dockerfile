# Render / cloud build — Docker context = repository root (.)
# Local build:  docker build -t annotra-api .
# (For backend-only context use backend/Dockerfile instead.)

FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "git+https://github.com/ultralytics/CLIP.git"

COPY backend/app ./app
COPY backend/data ./data
COPY backend/labels_65.txt ./labels_65.txt
COPY backend/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

COPY backend/best.pt backend/mobile_sam.pt ./

ENV MARINE_CUSTOM_YOLO=best.pt
ENV MARINE_SAM_MODEL=mobile_sam.pt
ENV MARINE_OPEN_VOCAB_MODEL=yolov8s-worldv2.pt
ENV MARINE_TFLITE_MODELS=
ENV MARINE_DEVICE=cpu
ENV MARINE_FAST_MODE_USE_TFLITE=false
ENV MARINE_DB_PATH=/data/annotations.db
ENV MARINE_DRIVE_CACHE_DIR=/data/drive-cache

EXPOSE 8765

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
