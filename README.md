# Marine Species Auto-Annotation Studio

Local-first, offline-capable desktop annotation platform for marine species — Electron + React + FastAPI with multi-model AI inference.

## Architecture

```
Electron Shell  →  React UI (Konva canvas)  →  FastAPI backend (YOLO / DINO / SAM / CLIP)
       ↓                      ↓                           ↓
 Local filesystem      WebSocket batch progress      SQLite + parallel workers
```

## Quick start

### 1. Install dependencies

```powershell
cd marine-annotation-studio
npm run install:all
```

### 2. Run in development

```powershell
npm run dev
```

This starts:
- FastAPI on `http://127.0.0.1:8765`
- Vite frontend on `http://127.0.0.1:5173`
- Electron desktop window

### 3. Production build

```powershell
cd frontend
npm run build
cd ..
npm start
```

## Features

- **Local folder picker** — index 100k+ images via Electron IPC (no uploads)
- **AI pipeline** — YOLOv8 → Grounding DINO → SAM 2 → CLIP → NMS fusion
- **Batch workers** — configurable thread pool with WebSocket progress
- **Konva canvas** — draw/edit boxes, zoom, pan, undo/redo
- **Export** — YOLO, COCO, Pascal VOC, CSV, labeled images
- **SQLite** — session persistence at `~/.marine-annotation-studio/annotations.db`
- **Fine-tuner** — train YOLOv8 on reviewed annotations via API

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| B | Draw box |
| V | Select |
| S | Smart box (SAM) |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| Ctrl+Shift+A | Auto-annotate image |
| Ctrl+Shift+B | Batch annotate |
| [ / ] | Prev / next image |
| F | Fit to screen |
| Del | Delete selected box |

## Custom marine model

Point to your trained weights in AI Settings or place `best.pt` and set `MARINE_CUSTOM_YOLO` env var.

Uses classes from `yolo_dataset/dataset.yaml` when creating a project from that folder.

## API docs

With backend running: [http://127.0.0.1:8765/docs](http://127.0.0.1:8765/docs)
