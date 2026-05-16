# Annotra — Deployment Guide (local & desktop)

Complete setup for **local development**, **production desktop (Electron)**, and **optional Google Drive** collaboration. Annotra is local-first: images stay on disk; the API runs on your machine.

> **Deploying to Vercel / Render / a cloud server?** See **[CLOUD-DEPLOYMENT.md](./CLOUD-DEPLOYMENT.md)** (split frontend + API, OAuth URLs, GPU options).

---

## 1. What you are deploying

| Component | Role | Default URL / path |
|-----------|------|-------------------|
| **FastAPI backend** | AI inference, SQLite DB, export, Drive OAuth | `http://127.0.0.1:8765` |
| **React frontend** | Annotra UI (Vite) | `http://127.0.0.1:5173` (dev) |
| **Electron** (optional) | Desktop shell, folder picker | Loads frontend + talks to backend |
| **SQLite** | Projects, images, annotations | `%USERPROFILE%\.marine-annotation-studio\annotations.db` |
| **Model weights** | `best.pt`, YOLO-World, SAM, TFLite | `marine-annotation-studio/backend/` |

```text
Electron / Browser  →  React (5173)  →  FastAPI (8765)  →  YOLO / World / SAM
                              ↓
                    Local image folders + SQLite
```

---

## 2. Prerequisites

### Software

- **Node.js** 18+ and **npm**
- **Python** 3.10–3.12 (3.13 works; first model load can be slow on CPU)
- **Git** (to clone the repo)
- **GPU (recommended)** — NVIDIA + CUDA for reasonable Auto/Batch speed; CPU works but is slow (1–3+ min per image with World enabled)

### Disk space (approximate)

| Asset | Size (order of magnitude) |
|-------|---------------------------|
| Python venv + PyTorch | 2–8 GB |
| `best.pt` (your model) | varies |
| `yolov8s-worldv2.pt` | ~50 MB (auto-download if missing) |
| `mobile_sam.pt` | ~40 MB |
| TFLite models (optional) | few MB each |
| Drive cache (optional) | grows with project size |

---

## 3. Clone and install

```powershell
cd D:\fish-sih-final\marine-annotation-studio
npm run install:all
```

This installs root + frontend npm packages and `pip install -r backend/requirements.txt`.

### Recommended: Python virtual environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use the same venv whenever you run uvicorn.

---

## 4. Model weights (required for AI)

Place files in **`backend/`** (short names are resolved automatically):

| File | Purpose | Required? |
|------|---------|-------------|
| `best.pt` | Primary species detector (your trained YOLO) | **Yes** for good results |
| `yolov8s-worldv2.pt` | YOLO-World open vocabulary | Recommended (Auto + Batch) |
| `mobile_sam.pt` | Smart segment tool | Optional |
| `model_nano32.tflite`, `model32.tflite` | Extra detectors | Optional |
| `species_crop_cls.pt` | Crop classifier | Optional |

Copy from your training output or SIH bundle. **Do not commit large `.pt` files to Git** (they are gitignored).

First backend start downloads missing Ultralytics weights if not present (needs internet once).

---

## 5. Environment variables (`backend/.env`)

Copy the example and edit:

```powershell
copy backend\.env.example backend\.env
```

All settings use the prefix **`MARINE_`**. The file lives at `backend/.env` only (never commit it).

### 5.1 Core detection

```env
# Primary model (filename in backend/ or absolute path)
MARINE_CUSTOM_YOLO=best.pt

# YOLO-World (open vocabulary)
MARINE_ENABLE_OPEN_VOCAB=true
MARINE_OPEN_VOCAB_MODEL=yolov8s-worldv2.pt
MARINE_OPEN_VOCAB_PROMPT_MODE=generic

# Inference
MARINE_CONFIDENCE=0.40
MARINE_MIN_DETECTION_CONFIDENCE=0.40
MARINE_OPEN_VOCAB_MIN_CONFIDENCE=0.12
MARINE_MAX_BOX_AREA_RATIO=0.55
MARINE_IMGSZ=1280
MARINE_MAX_BOXES=100

# GPU: auto | cpu | cuda:0 | mps
MARINE_DEVICE=cuda:0
MARINE_HALF_PRECISION=true
```

### 5.2 Auto vs Batch

```env
MARINE_AUTO_USE_ALL_MODELS=true
MARINE_FAST_MODE_USE_TFLITE=false
MARINE_ENABLE_SUBJECT_BOX_EXPAND=false
MARINE_FOLDER_LABEL_MODE=smart

MARINE_BATCH_ACCURATE_MODE=true
MARINE_BATCH_USE_OPEN_VOCAB=true
MARINE_BATCH_IMGSZ=640
MARINE_BATCH_ACCURATE_IMGSZ=640
MARINE_BATCH_OPEN_VOCAB_IMGSZ=960
MARINE_BATCH_INFERENCE_SIZE=8
```

- **Batch** runs `best.pt` + **YOLO-World** per image (progress 1/4, 2/4, …).
- First image on **CPU** can take **1–3+ minutes**; use GPU when possible.

### 5.3 SAM / segment tools

```env
MARINE_ENABLE_SAM_SEGMENT=true
MARINE_SAM_MODEL=mobile_sam.pt
MARINE_MAGIC_WAND_TOLERANCE=35
```

### 5.4 Optional TFLite

```env
MARINE_TFLITE_MODELS=model_nano32.tflite,model32.tflite
```

### 5.5 Database and cache paths (optional overrides)

```env
# Defaults — usually leave unset
# MARINE_DB_PATH=C:\Users\You\.marine-annotation-studio\annotations.db
# MARINE_DRIVE_CACHE_DIR=C:\Users\You\.marine-annotation-studio\drive-cache
```

---

## 6. API keys and external services

### Required for core annotation (local folders)

**No API keys.** Everything runs offline after models are downloaded.

### Optional: Google Drive projects

Needed only if collaborators use **Google Drive** as the dataset source.

| Secret | Where to get it |
|--------|-----------------|
| `MARINE_GOOGLE_CLIENT_ID` | [Google Cloud Console](https://console.cloud.google.com/) |
| `MARINE_GOOGLE_CLIENT_SECRET` | Same OAuth 2.0 **Web application** client |

#### Google Cloud setup (step by step)

1. Create a project → **APIs & Services** → **Enable** “Google Drive API”.
2. **Credentials** → **Create credentials** → **OAuth client ID** → **Web application**.
3. **Authorized redirect URIs** (must match exactly):
   - `http://127.0.0.1:8765/drive/oauth/callback`
4. Copy **Client ID** and **Client secret** into `backend/.env`:

```env
MARINE_GOOGLE_CLIENT_ID=123456789-xxxx.apps.googleusercontent.com
MARINE_GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxx
MARINE_GOOGLE_OAUTH_REDIRECT_URI=http://127.0.0.1:8765/drive/oauth/callback
MARINE_GOOGLE_OAUTH_FRONTEND_REDIRECT=http://127.0.0.1:5173
MARINE_DRIVE_CACHE_ENABLED=true
```

5. **OAuth consent screen**: add test users while in “Testing”, or publish for production.
6. In Annotra: **Connect Google Drive** on the dashboard; browser completes OAuth and returns to the app.

**Production Electron note:** If the UI is not on port 5173, set `MARINE_GOOGLE_OAUTH_FRONTEND_REDIRECT` to your actual frontend origin.

### What is NOT used

- No OpenAI / Anthropic keys
- No cloud annotation API (unless you add one later)
- No AWS S3 keys in the stock app

---

## 7. Run locally (development)

### Terminal A — backend

```powershell
cd marine-annotation-studio
npm run dev:backend
```

Wait until you see:

```text
Uvicorn running on http://127.0.0.1:8765
```

Verify: [http://127.0.0.1:8765/health](http://127.0.0.1:8765/health) → `yolo_status: "ready"`.

If port 8765 is busy:

```powershell
npm run kill:backend
npm run dev:backend
```

### Terminal B — frontend

```powershell
cd marine-annotation-studio\frontend
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

### Full stack + Electron

```powershell
cd marine-annotation-studio
npm run dev
```

Starts backend, Vite, and Electron together.

---

## 8. Production desktop deployment

### Build frontend

```powershell
cd marine-annotation-studio
npm run build
```

Output: `frontend/dist/`.

### Run packaged-style locally

```powershell
npm start
```

Electron loads `frontend/dist` and **spawns** the Python backend on port **8765** (see `electron/main.cjs`). Ensure:

- `python` is on PATH
- `backend/.env` exists
- Weights are in `backend/`

### Distributing to another PC

Ship this folder (or installer you build):

```text
marine-annotation-studio/
  backend/          # code + .env + *.pt / *.tflite
  frontend/dist/    # after npm run build
  electron/
  package.json
```

On the target machine:

1. Install Node, Python, dependencies (`npm run install:all`).
2. Copy `backend/.env` and model weights (secure channel).
3. `npm start`.

For a proper installer, use **electron-builder** (not included by default); point it at the same layout.

---

## 9. Deploying backend on a server (advanced)

The UI defaults to `http://127.0.0.1:8765` in `frontend/src/api.ts`. For a remote API:

1. Run uvicorn bound to the server:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8765
```

2. Change `BASE` in `frontend/src/api.ts` (or add `VITE_API_URL` and read `import.meta.env.VITE_API_URL`).
3. Rebuild frontend.
4. Put **HTTPS** and a reverse proxy (nginx) in front for production.
5. Update Google OAuth redirect URIs to your public backend URL.

**Security:** Do not expose the API without authentication; the stock app is designed for localhost.

---

## 10. Ports and firewall

| Port | Service |
|------|---------|
| 8765 | FastAPI (health, annotate, batch, Drive OAuth callback) |
| 5173 | Vite dev server only |

Allow localhost; open inbound 8765 only if you intentionally run a network deployment.

---

## 11. Health check and troubleshooting

| Symptom | Fix |
|---------|-----|
| `ERR_CONNECTION_REFUSED` on `/health` | Start backend; wait for model load |
| `npm run dev:backend` hangs | Use direct uvicorn (see §7); avoid stuck `kill-port` |
| Auto/Batch disabled | Wait until `/health` shows `yolo_status: "ready"` |
| Batch stuck at `0/4` | Normal on CPU for first image; wait 1–3 min |
| `boolean index` / model errors | Re-download corrupt `mobile_sam.pt` or `best.pt` |
| Drive batch fails | Set Google OAuth vars; or **Link local folder** for same dataset |
| Huge bounding boxes | `MARINE_ENABLE_SUBJECT_BOX_EXPAND=false`, `MARINE_FOLDER_LABEL_MODE=smart` |

Useful endpoints:

- Health: `GET /health`
- OpenAPI: `GET /docs`

---

## 12. Security checklist

- [ ] Never commit `backend/.env` or OAuth secrets
- [ ] Restrict Google OAuth client to your redirect URIs
- [ ] Keep `best.pt` and datasets on trusted machines
- [ ] Do not expose port 8765 to the public internet without auth + TLS
- [ ] Rotate `MARINE_GOOGLE_CLIENT_SECRET` if leaked

---

## 13. Quick reference — minimum `.env` for a new machine

```env
MARINE_CUSTOM_YOLO=best.pt
MARINE_SAM_MODEL=mobile_sam.pt
MARINE_ENABLE_OPEN_VOCAB=true
MARINE_OPEN_VOCAB_MODEL=yolov8s-worldv2.pt
MARINE_BATCH_USE_OPEN_VOCAB=true
MARINE_DEVICE=cuda:0

# Optional Drive only:
# MARINE_GOOGLE_CLIENT_ID=...
# MARINE_GOOGLE_CLIENT_SECRET=...
```

---

## 14. Support workflow

1. Backend running + `/health` → `ready`
2. Open local project folder (species subfolders improve labels)
3. Test **Auto** on one image
4. **Batch** → OK = unannotated only, Cancel = all images
5. Export via Export panel (YOLO / COCO / etc.)

For SIH / fish dataset work, keep species images under folders like `Green_Chromide/photo.jpg` rather than only `Uncategorized/`.
