# Annotra — Cloud deployment guide (Vercel, Render, and alternatives)

Annotra is a **PyTorch + YOLO** app, not a simple CRUD site. You must split it into:

| Part | What it is | Best hosts |
|------|------------|------------|
| **Frontend** | Static React (Vite) | **Vercel**, Netlify, Cloudflare Pages |
| **Backend API** | FastAPI + large ML models | **Render**, Railway, Fly.io, **GPU VM** |
| **Desktop** | Electron + local folders | User’s PC (not Vercel/Render) |

**Do not put the ML backend on Vercel** — serverless limits (memory, time, no GPU) make Auto/Batch impractical.

---

## Recommended architecture (best fit)

```text
                    ┌─────────────────────┐
  Users (browser)   │  Vercel / Netlify   │  annotra.vercel.app
                    │  (static frontend)  │
                    └──────────┬──────────┘
                               │ HTTPS  VITE_API_URL
                               ▼
                    ┌─────────────────────┐
                    │  Render / Railway   │  annotra-api.onrender.com
                    │  Docker + FastAPI   │  4–8 GB RAM minimum
                    │  + persistent disk  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        Google Drive      SQLite on disk    best.pt + World
        (optional OAuth)  (/data volume)     baked or mounted
```

**Best for your SIH project today**

| Goal | Recommendation |
|------|----------------|
| Demos / team in browser with **Drive** datasets | **Vercel (UI) + Render (API)** |
| Fast Auto/Batch on many images | **GPU VM** (AWS g4dn, Lambda Labs, RunPod) — not free tier |
| Annotators on lab PCs with local folders | **Keep Electron + local backend** ([DEPLOYMENT.md](./DEPLOYMENT.md)) |

---

## Platform comparison

| Platform | Frontend | ML backend | GPU | Verdict |
|----------|----------|------------|-----|---------|
| **Vercel** | Excellent | No | No | UI only |
| **Render** | Static site OK | Docker web service + disk | No (CPU only) | Good starter API |
| **Railway** | Static | Docker, volumes | No | Similar to Render |
| **Fly.io** | Static | Docker, volumes | No | Good; global regions |
| **Netlify** | Excellent | No | No | UI only |
| **AWS EC2 / DO Droplet** | Optional | Full control | Yes (extra $) | **Best performance** |
| **Hugging Face Spaces** | Space UI | Gradio/API | Optional GPU | Demo only, not full app |

**Minimum API server:** **4 GB RAM** (tight), **8 GB RAM** recommended for `best.pt` + YOLO-World on CPU.

---

## What changes in the cloud vs desktop

| Desktop (current) | Cloud (you must accept) |
|-------------------|-------------------------|
| Open folder on disk | **Google Drive** projects or future S3 upload |
| `127.0.0.1:8765` | Public HTTPS API URL |
| SQLite in user profile | SQLite on **persistent volume** (Render disk) or PostgreSQL (future) |
| Electron file picker | Browser only |

Local-folder batch only works when the **backend can read those paths** (same machine). Remote annotators should use **Drive** or you host data on the server.

---

## Step 1 — Bundle model weights in Docker

Weights are **baked into the API image** (not downloaded from Git — `.pt` files are too large for GitHub).

### Files in `backend/`

| File | Role |
|------|------|
| `best.pt` | **Required** — copy into image at build time |
| `mobile_sam.pt` | Recommended — Smart segment |
| `yolov8s-worldv2.pt` | Optional in image; **auto-downloaded on first container start** if missing |
| `model_nano32.tflite`, `model32.tflite` | Optional — included if present in `backend/` |

### Check before build (Windows)

```powershell
cd marine-annotation-studio\backend
.\scripts\prepare-docker-build.ps1
```

### Build image locally

```powershell
cd marine-annotation-studio\backend
docker build -t annotra-api .
docker run --rm -p 8765:8765 -v annotra-data:/data annotra-api
```

Open `http://127.0.0.1:8765/health` — wait until `yolo_status` is `"ready"`.

### Deploy to Render from GitHub

Git does **not** include `best.pt`. Use one of:

1. **Build locally and push** to GitHub Container Registry / Docker Hub, then point Render at that image, **or**
2. **Render manual deploy** from a branch that uses **Git LFS** for `backend/*.pt`, **or**
3. Add a private S3 URL and extend the Dockerfile with `curl` in build (team-specific).

**Simplest for SIH:** build on your PC, push image:

```powershell
docker build -t annotra-api .\backend
docker tag annotra-api ghcr.io/YOUR_USER/annotra-api:latest
docker push ghcr.io/YOUR_USER/annotra-api:latest
```

On Render: Web Service → **Existing image** → paste `ghcr.io/YOUR_USER/annotra-api:latest`.

### Repo files for cloud

- `Dockerfile` (repo root) — Render build; copies `backend/best.pt`, `backend/mobile_sam.pt`
- `backend/Dockerfile` — local build when context is `backend/`
- `backend/docker-entrypoint.sh` — verifies `best.pt`, downloads World model if needed
- `render.yaml` — Render blueprint
- `frontend/vercel.json` — SPA routing
- `VITE_API_URL` — points UI at your API (see `frontend/src/apiBase.ts`)

---

## Step 2 — Deploy API on Render

### A. Create Web Service (Docker)

1. Push repo to **GitHub**.
2. [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint** (uses `render.yaml`)  
   **or** **New Web Service** → connect repo → **Docker**, path `Dockerfile`, context `.` (repo root).  
   If you use `backend/Dockerfile` + context `backend`, weights must sit in `backend/` (not repo root).
3. **Plan:** at least **Standard** (2 GB+ RAM). Upgrade if OOM during model load.
4. **Disk:** add **20 GB** persistent disk, mount `/data` (matches `render.yaml`).
5. **Environment variables** (sensitive → Secret):

```env
MARINE_DB_PATH=/data/annotations.db
MARINE_DRIVE_CACHE_DIR=/data/drive-cache
MARINE_DEVICE=cpu
MARINE_CUSTOM_YOLO=best.pt
MARINE_ENABLE_OPEN_VOCAB=true
MARINE_OPEN_VOCAB_MODEL=yolov8s-worldv2.pt
MARINE_BATCH_USE_OPEN_VOCAB=true

# Google Drive (required for Drive projects in cloud)
MARINE_GOOGLE_CLIENT_ID=<from Google Cloud>
MARINE_GOOGLE_CLIENT_SECRET=<secret>
MARINE_GOOGLE_OAUTH_REDIRECT_URI=https://YOUR-API.onrender.com/drive/oauth/callback
MARINE_GOOGLE_OAUTH_FRONTEND_REDIRECT=https://YOUR-APP.vercel.app
```

6. Deploy. First boot may take **5–15 minutes** (install + model load).
7. Note public URL: `https://annotra-api-xxxx.onrender.com`
8. Test: `https://YOUR-API.onrender.com/health` → `"yolo_status":"ready"`.

### B. Render limitations

- **CPU only** on standard plans → slow Auto/Batch (same as local CPU).
- **Cold start** on free/starter—service sleeps; first request slow.
- **Request timeout** 30–100 min depending on plan—long batch jobs should use background tasks (already implemented).

---

## Step 3 — Deploy frontend on Vercel

1. [Vercel](https://vercel.com/) → **Add New Project** → import Git repo.
2. **Root Directory:** `marine-annotation-studio/frontend`
3. **Framework:** Vite  
   **Build command:** `npm run build`  
   **Output:** `dist`
4. **Environment variables** (Production):

```env
VITE_API_URL=https://YOUR-API.onrender.com
```

5. Deploy. Open `https://your-app.vercel.app`.

### Google OAuth (cloud)

In Google Cloud Console → OAuth client → **Authorized redirect URIs:**

- `https://YOUR-API.onrender.com/drive/oauth/callback`

**Authorized JavaScript origins:**

- `https://your-app.vercel.app`

Match `MARINE_GOOGLE_OAUTH_*` on the API exactly.

---

## Step 4 — Railway (alternative to Render)

1. [Railway](https://railway.app/) → New Project → **Deploy from GitHub**.
2. Add service → **Dockerfile** path `Dockerfile`, root directory = repo root (or `backend/Dockerfile` + root `backend`).
3. Add **Volume** mounted at `/data`; set `MARINE_DB_PATH=/data/annotations.db`.
4. Set same env vars as Render.
5. Generate domain → use as `VITE_API_URL` on Vercel.

Railway pricing is usage-based; ML workloads need **8 GB** service tier.

---

## Step 5 — Fly.io (alternative)

```bash
cd marine-annotation-studio/backend
fly launch --no-deploy
fly volumes create annotra_data --size 10
# attach volume to fly.toml mount /data
fly secrets set MARINE_GOOGLE_CLIENT_ID=... MARINE_GOOGLE_CLIENT_SECRET=...
fly deploy
```

Set `VITE_API_URL=https://your-app.fly.dev` on Vercel.

---

## Step 6 — Production GPU (when Render CPU is too slow)

Use a **VM with NVIDIA GPU**:

| Provider | Notes |
|----------|--------|
| AWS EC2 `g4dn.xlarge` | CUDA, full control |
| Google Cloud GCE + T4 | Same |
| Lambda Labs / RunPod | Cheaper hourly GPU |
| Azure NC-series | Enterprise |

On the VM:

```bash
git clone <repo>
cd marine-annotation-studio/backend
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# copy .env + weights
MARINE_DEVICE=cuda:0 uvicorn app.main:app --host 0.0.0.0 --port 8765
```

Put **nginx** + **Let’s Encrypt** in front; point `VITE_API_URL` to `https://api.yourdomain.com`.

---

## API keys & secrets checklist (cloud)

| Variable | Where | Required? |
|----------|--------|-----------|
| `MARINE_GOOGLE_CLIENT_ID` | Render/Railway secrets | Only for Drive |
| `MARINE_GOOGLE_CLIENT_SECRET` | Render secrets | Only for Drive |
| `MARINE_GOOGLE_OAUTH_REDIRECT_URI` | API public URL + `/drive/oauth/callback` | Drive |
| `MARINE_GOOGLE_OAUTH_FRONTEND_REDIRECT` | Vercel URL | Drive |
| `VITE_API_URL` | Vercel env | **Yes** for hosted UI |
| OpenAI / other AI keys | — | **Not used** |

No other third-party API keys are required for core YOLO annotation.

---

## Security (public deployment)

- [ ] Use **HTTPS** only (Vercel + Render provide it).
- [ ] Do not commit `.env` or OAuth secrets.
- [ ] Restrict Google OAuth redirect URIs to your domains.
- [ ] Add **API auth** before exposing to the open internet (stock app has no login—anyone with the URL can call `/annotate`).
- [ ] Consider IP allowlist or VPN for team-only API.

---

## Quick deploy checklist

1. [ ] Models in `backend/` or download on startup  
2. [ ] Deploy **Docker API** on Render (8 GB RAM if possible) + disk `/data`  
3. [ ] `/health` returns `ready`  
4. [ ] Set Google OAuth URLs to production domains  
5. [ ] Deploy **frontend** on Vercel with `VITE_API_URL`  
6. [ ] Connect Drive in UI → create/open Drive project  
7. [ ] Test **Auto** on one image, then **Batch**

---

## When to stay local (Electron)

Keep using [DEPLOYMENT.md](./DEPLOYMENT.md) if:

- Images are only on annotators’ hard drives  
- You need maximum speed (local GPU)  
- You do not want to pay for cloud RAM  
- SIH judging is offline

**Hybrid:** Developers use Electron locally; reviewers use **Vercel + Render + Drive**.

---

## Troubleshooting cloud

| Issue | Fix |
|-------|-----|
| `dockerDesktopLinuxEngine` / pipe not found | **Start Docker Desktop** on Windows; wait until it says “Running”, then retry `docker build` |
| `Cannot find command 'git'` / CLIP install fails | Rebuild image with latest `backend/Dockerfile` (installs `git` + CLIP at build time) |
| `set_classes failed (No module named 'clip')` | Same — rebuild Docker image; do not rely on runtime `pip install` |
| `"/mobile_sam.pt": not found` during Docker build | Render **Docker context** must be repo root (`.`) with **`Dockerfile`** at root, not `backend/Dockerfile` with context `.` — or set context to `backend` to match `backend/Dockerfile` |
| `SAM load failed` / corrupt checkpoint | `mobile_sam.pt` truncated — re-download or re-copy from PC, then `docker build` again |
| UI “Backend offline” | Wrong `VITE_API_URL`; rebuild Vercel after changing env |
| CORS errors | Backend allows `*`; ensure API URL has no trailing slash |
| OAuth redirect mismatch | Google Console URIs must match `.env` exactly |
| 502 / OOM on Render | Upgrade RAM; set `MARINE_FAST_MODE_USE_TFLITE=false` |
| Batch never finishes | CPU timeout—increase plan or use GPU VM |
| “Link local folder” useless in browser | Expected—use Drive or deploy desktop app |

---

## Summary

| Deploy | Use for |
|--------|---------|
| **Vercel + Render** | Best **browser** setup for Drive-based teams |
| **Render API only** | API for custom frontends |
| **GPU VM** | Best **speed** for Auto/Batch |
| **Electron local** | Best **local folder** workflow |

Start with **Render (API) + Vercel (UI) + Google Drive**; move to a **GPU VM** when CPU inference is too slow.
