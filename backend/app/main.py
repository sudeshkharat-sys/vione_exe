import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from .database import init_db, async_session
from .api import projects, images, annotations, pipeline, auth, videos
from .config import settings
from .models.training_job import TrainingJob
from sqlalchemy import select, update
from datetime import datetime


def _find_frontend_build() -> Path | None:
    """Locate the React build directory in both Docker and EXE modes."""
    candidates = [
        # EXE mode: PyInstaller extracts to _MEIPASS/frontend_build
        Path(getattr(sys, "_MEIPASS", "")) / "frontend_build",
        # Docker / dev: build is placed next to the backend package
        Path(__file__).parent.parent / "frontend" / "build",
    ]
    for p in candidates:
        if p.is_dir() and (p / "index.html").exists():
            return p
    return None


async def _purge_stale_tasks():
    """On startup: purge queued Celery tasks and mark all pending/started DB jobs as failed.

    Any tasks left in the broker from a previous session would otherwise run
    silently in the background with no way to track or cancel them from the UI.
    """
    from .tasks.celery_app import celery_app

    # 1. Purge all tasks waiting in the broker queue (not yet picked up by a worker)
    try:
        celery_app.control.purge()
    except Exception:
        pass

    # 2. Mark every non-terminal DB job as failed so the UI shows a clean slate
    try:
        async with async_session() as session:
            await session.execute(
                update(TrainingJob)
                .where(TrainingJob.status.in_(["pending", "started"]))
                .values(
                    status="failure",
                    finished_at=datetime.utcnow(),
                    result_meta={"error": "App restarted — task was purged"},
                )
            )
            await session.commit()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await _purge_stale_tasks()
    yield

app = FastAPI(title="AI Vision Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(images.router, prefix="/api/v1")
app.include_router(annotations.router, prefix="/api/v1")
app.include_router(pipeline.router, prefix="/api/v1")
app.include_router(videos.router, prefix="/api/v1")

app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")


# In EXE mode (or when a frontend build is present) serve the React SPA.
# Docker mode: Nginx handles this — the mount is skipped when no build exists.
_frontend_dir = _find_frontend_build()
if os.environ.get("EXE_MODE") == "true" and _frontend_dir:
    # Serve React static assets (JS/CSS/images)
    app.mount("/static", StaticFiles(directory=str(_frontend_dir / "static")), name="react-static")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # SPA catch-all: every non-API path returns index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index = _frontend_dir / "index.html"
        return FileResponse(str(index))

else:
    @app.get("/health")
    async def health():
        return {"status": "ok"}
