import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import router

app = FastAPI(title="Annotra API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    init_db()
    from app.model_ensemble import ensemble

    ensemble.preload_async()

    from app.config import settings

    if settings.enable_sam_segment:

        def _warm_sam() -> None:
            try:
                from app.segmentation import _get_sam_model

                _get_sam_model()
            except Exception:
                pass

        threading.Thread(target=_warm_sam, name="sam-warmup", daemon=True).start()


@app.get("/")
def root():
    return {"app": "Annotra", "tagline": "Precision annotation for ML datasets", "docs": "/docs"}
