from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import base64, io, cv2, numpy as np
from ..tasks.training import train_seed_model, train_main_model
from ..tasks.auto_annotate import auto_annotate_remaining
from ..tasks.ai_prompt import detect_with_prompt, bulk_detect_with_prompt
from ..tasks.active_learning import (
    score_unlabeled_images,
    curriculum_auto_annotate,
    suggest_for_review,
)
from ..tasks.celery_app import celery_app
from ..database import get_db
from ..models.image import Image
from ..models.annotation import Annotation
from ..models.project import Project
from ..models.training_job import TrainingJob
from ..models.user import User
from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from celery.result import AsyncResult
from ..config import settings

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ── Available YOLO models ─────────────────────────────────────────

YOLO_MODELS = [
    # YOLO26 (latest Ultralytics — edge-optimised, NMS-free)
    {"value": "yolo26n.pt", "label": "YOLO26 Nano — fastest edge",    "family": "YOLO26"},
    {"value": "yolo26s.pt", "label": "YOLO26 Small",                  "family": "YOLO26"},
    {"value": "yolo26m.pt", "label": "YOLO26 Medium",                 "family": "YOLO26"},
    {"value": "yolo26l.pt", "label": "YOLO26 Large",                  "family": "YOLO26"},
    {"value": "yolo26x.pt", "label": "YOLO26 XL — best accuracy",    "family": "YOLO26"},
    # YOLO12 (attention-centric, NeurIPS 2025)
    {"value": "yolo12n.pt", "label": "YOLO12 Nano",                   "family": "YOLO12"},
    {"value": "yolo12s.pt", "label": "YOLO12 Small",                  "family": "YOLO12"},
    {"value": "yolo12m.pt", "label": "YOLO12 Medium",                 "family": "YOLO12"},
    {"value": "yolo12l.pt", "label": "YOLO12 Large",                  "family": "YOLO12"},
    {"value": "yolo12x.pt", "label": "YOLO12 XL",                     "family": "YOLO12"},
    # YOLO11
    {"value": "yolo11n.pt", "label": "YOLO11 Nano — fastest",        "family": "YOLO11"},
    {"value": "yolo11s.pt", "label": "YOLO11 Small",                  "family": "YOLO11"},
    {"value": "yolo11m.pt", "label": "YOLO11 Medium",                 "family": "YOLO11"},
    {"value": "yolo11l.pt", "label": "YOLO11 Large",                  "family": "YOLO11"},
    {"value": "yolo11x.pt", "label": "YOLO11 XL — best accuracy",    "family": "YOLO11"},
    # YOLOv10
    {"value": "yolov10n.pt", "label": "YOLOv10 Nano",                 "family": "YOLOv10"},
    {"value": "yolov10s.pt", "label": "YOLOv10 Small",                "family": "YOLOv10"},
    {"value": "yolov10m.pt", "label": "YOLOv10 Medium",               "family": "YOLOv10"},
    {"value": "yolov10b.pt", "label": "YOLOv10 Base",                 "family": "YOLOv10"},
    {"value": "yolov10l.pt", "label": "YOLOv10 Large",                "family": "YOLOv10"},
    {"value": "yolov10x.pt", "label": "YOLOv10 XL",                   "family": "YOLOv10"},
    # YOLOv9
    {"value": "yolov9c.pt", "label": "YOLOv9 C",                      "family": "YOLOv9"},
    {"value": "yolov9e.pt", "label": "YOLOv9 E — high accuracy",      "family": "YOLOv9"},
    # YOLOv8
    {"value": "yolov8n.pt", "label": "YOLOv8 Nano",                   "family": "YOLOv8"},
    {"value": "yolov8s.pt", "label": "YOLOv8 Small",                  "family": "YOLOv8"},
    {"value": "yolov8m.pt", "label": "YOLOv8 Medium",                 "family": "YOLOv8"},
    {"value": "yolov8l.pt", "label": "YOLOv8 Large",                  "family": "YOLOv8"},
    {"value": "yolov8x.pt", "label": "YOLOv8 XL",                     "family": "YOLOv8"},
]


@router.get("/available-models")
async def get_available_models():
    """Return the list of supported YOLO model weights for the UI dropdowns.

    Each model entry now includes `available` (True when the weight is
    pre-downloaded and offline-ready) and `size_mb` (file size in MB or None).
    """
    weights_dir = settings.yolo_weights_dir
    families: dict = {}
    models_out = []
    for m in YOLO_MODELS:
        path = weights_dir / m["value"]
        available = path.exists() and path.stat().st_size > 1024 * 1024
        entry = {
            **m,
            "available": available,
            "size_mb": round(path.stat().st_size / 1_048_576, 1) if available else None,
        }
        models_out.append(entry)
        families.setdefault(m["family"], []).append(
            {"value": m["value"], "label": m["label"],
             "available": available, "size_mb": entry["size_mb"]}
        )
    return {"models": models_out, "families": families}


@router.get("/weights-status")
async def get_weights_status(
    current_user: User = Depends(get_current_user),
):
    """
    Report which YOLO base weights are pre-downloaded and offline-ready.

    Returns a per-model breakdown plus a summary so the frontend can show
    a simple indicator (all green / some missing / none pre-loaded).
    """
    weights_dir = settings.yolo_weights_dir
    models_out = []
    total_size_mb = 0.0

    for m in YOLO_MODELS:
        path = weights_dir / m["value"]
        available = path.exists() and path.stat().st_size > 1024 * 1024
        size_mb = round(path.stat().st_size / 1_048_576, 1) if available else None
        if size_mb:
            total_size_mb += size_mb
        models_out.append({
            **m,
            "available": available,
            "size_mb": size_mb,
            "path": str(path) if available else None,
        })

    n_available = sum(1 for m in models_out if m["available"])
    return {
        "weights_dir": str(weights_dir),
        "total_models": len(YOLO_MODELS),
        "available_count": n_available,
        "total_size_mb": round(total_size_mb, 1),
        "offline_ready": n_available == len(YOLO_MODELS),
        "models": models_out,
    }


# ── Training ──────────────────────────────────────────────────────

class TrainSeedRequest(BaseModel):
    model_name: str = "yolo11s.pt"
    epochs: int = 100
    imgsz: int = 640
    preprocess: bool = True
    batch: int = -1


@router.post("/train-seed/{project_id}")
async def start_seed_training(
    project_id: str,
    body: TrainSeedRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)
    req = body or TrainSeedRequest()
    task = train_seed_model.delay(
        project_id, req.model_name, req.epochs, req.imgsz, req.preprocess, req.batch
    )
    return {"task_id": task.id, "status": "queued"}


class TrainMainRequest(BaseModel):
    model_name: str = "yolo11s.pt"
    epochs: int = 150
    use_seed_weights: bool = True
    imgsz: int = 640
    preprocess: bool = True
    batch: int = -1


@router.post("/train-main/{project_id}")
async def start_main_training(
    project_id: str,
    body: TrainMainRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)
    req = body or TrainMainRequest()
    task = train_main_model.delay(
        project_id, req.model_name, req.epochs,
        req.use_seed_weights, req.imgsz, req.preprocess, req.batch
    )
    return {"task_id": task.id, "status": "queued"}


@router.get("/clahe-preview/{project_id}")
async def get_clahe_preview(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return a before/after CLAHE preview using the first annotated image in the
    project.  Both images are returned as base64-encoded JPEG data URIs.
    """
    await get_owned_project(project_id, current_user, db)

    result = await db.execute(
        select(Image)
        .where(Image.project_id == project_id, Image.status == "annotated")
        .limit(1)
    )
    img_row = result.scalar_one_or_none()
    if img_row is None:
        result = await db.execute(
            select(Image).where(Image.project_id == project_id).limit(1)
        )
        img_row = result.scalar_one_or_none()

    if img_row is None:
        raise HTTPException(status_code=404, detail="No images found in project")

    rel = img_row.filepath.lstrip("/")
    candidates = [
        settings.upload_dir.resolve().parent / rel,
        settings.upload_dir.resolve() / rel,
    ]
    file_path = next((p for p in candidates if p.exists()), None)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    img_bgr = cv2.imread(str(file_path))
    if img_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    h, w = img_bgr.shape[:2]
    max_w = 480
    if w > max_w:
        scale = max_w / w
        img_bgr = cv2.resize(img_bgr, (max_w, int(h * scale)))

    def to_data_uri(bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(status_code=500, detail="Image encoding failed")
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge([clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    lut = np.array([(i / 255.0) ** 1.3 * 255 for i in range(256)], dtype=np.uint8)
    enhanced = cv2.LUT(enhanced, lut)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    enhanced = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

    return {
        "filename": img_row.filename,
        "original": to_data_uri(img_bgr),
        "enhanced": to_data_uri(enhanced),
    }


# ── Auto-annotate ────────────────────────────────────────────────

class AutoAnnotateRequest(BaseModel):
    image_ids: Optional[List[str]] = None
    conf: float = 0.25
    use_tta: bool = False


@router.post("/auto-annotate/{project_id}")
async def start_auto_annotation(
    project_id: str,
    body: AutoAnnotateRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)
    ids     = body.image_ids if body else None
    conf    = body.conf if body else 0.25
    use_tta = body.use_tta if body else False
    task = auto_annotate_remaining.delay(project_id, ids, conf, use_tta)
    return {"task_id": task.id, "status": "queued"}


# ── AI Prompt ───────────────────────────────────────────────────

class AIPromptRequest(BaseModel):
    project_id: str
    image_id: str
    prompt: str
    clear_existing: bool = False


class AIBulkPromptRequest(BaseModel):
    project_id: str
    prompt: str
    image_ids: Optional[List[str]] = None


@router.post("/ai-prompt")
async def trigger_ai_prompt(
    body: AIPromptRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Detect objects in a single image using a text prompt."""
    await get_owned_project(body.project_id, current_user, db)
    task = detect_with_prompt.delay(
        body.project_id, body.image_id, body.prompt, body.clear_existing
    )
    return {"task_id": task.id}


@router.post("/ai-bulk-prompt")
async def trigger_ai_bulk_prompt(
    body: AIBulkPromptRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Detect objects in multiple images using a text prompt."""
    await get_owned_project(body.project_id, current_user, db)
    task = bulk_detect_with_prompt.delay(
        body.project_id, body.prompt, body.image_ids
    )
    return {"task_id": task.id}


@router.get("/pending-images/{project_id}")
async def get_pending_images(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return pending images AND annotated-but-empty images."""
    await get_owned_project(project_id, current_user, db)

    pending_q = await db.execute(
        select(Image).where(Image.project_id == project_id, Image.status == "pending")
    )
    pending_images = pending_q.scalars().all()

    annotated_q = await db.execute(
        select(Image).where(Image.project_id == project_id, Image.status == "annotated")
    )
    all_annotated = annotated_q.scalars().all()

    empty_annotated = []
    for img in all_annotated:
        count_q = await db.execute(
            select(func.count(Annotation.id)).where(Annotation.image_id == img.id)
        )
        count = count_q.scalar()
        if count == 0:
            img.status = "pending"
            empty_annotated.append(img)

    if empty_annotated:
        await db.commit()

    images = pending_images + empty_annotated
    return [
        {
            "id": img.id,
            "filename": img.filename,
            "filepath": img.filepath,
            "width": img.width,
            "height": img.height,
        }
        for img in images
    ]


@router.get("/model-status/{project_id}")
async def get_model_status(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Check whether trained seed/main models exist for this project."""
    await get_owned_project(project_id, current_user, db)
    seed_path = settings.model_dir / project_id / "seed_best.pt"
    main_path = settings.model_dir / project_id / "main_best.pt"
    return {
        "has_seed_model":  seed_path.exists(),
        "model_path":      str(seed_path) if seed_path.exists() else None,
        "seed_model_path": str(seed_path) if seed_path.exists() else None,
        "has_main_model":  main_path.exists(),
        "main_model_path": str(main_path) if main_path.exists() else None,
    }


@router.get("/model-details/{project_id}")
async def get_model_details(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return rich details about trained models for a project."""
    await get_owned_project(project_id, current_user, db)

    seed_path = settings.model_dir / project_id / "seed_best.pt"
    main_path = settings.model_dir / project_id / "main_best.pt"

    def file_info(path):
        if not path.exists():
            return {"exists": False, "file_size_mb": None, "modified_at": None}
        stat = path.stat()
        return {
            "exists": True,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    async def latest_job(job_type: str):
        q = (
            select(TrainingJob)
            .where(TrainingJob.project_id == project_id, TrainingJob.job_type == job_type)
            .order_by(desc(TrainingJob.created_at))
            .limit(1)
        )
        result = await db.execute(q)
        job = result.scalar_one_or_none()
        if not job:
            return None
        return {
            "id": job.id,
            "status": job.status,
            "result_meta": job.result_meta or {},
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }

    return {
        "seed": {**file_info(seed_path), "last_job": await latest_job("seed_training")},
        "main": {**file_info(main_path), "last_job": await latest_job("main_training")},
    }


@router.get("/download-model/{project_id}/{model_type}")
async def download_model(
    project_id: str,
    model_type: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream the trained model weights file as a download."""
    await get_owned_project(project_id, current_user, db)

    if model_type not in ("seed", "main"):
        raise HTTPException(status_code=400, detail="model_type must be 'seed' or 'main'")

    filename = f"{model_type}_best.pt"
    path = settings.model_dir / project_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{model_type} model not found for this project")

    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=filename,
    )


# ── Active Learning ──────────────────────────────────────────────

class ScoreImagesRequest(BaseModel):
    model_type: str = "seed"
    strategy: str = "combined"
    top_k: int = 0
    use_tta: bool = False


class CurriculumAnnotateRequest(BaseModel):
    high_conf: float = 0.6
    low_conf: float = 0.25
    review_band_top: float = 0.6
    review_band_bottom: float = 0.35
    use_tta: bool = True


class SuggestReviewRequest(BaseModel):
    budget: int = 10
    strategy: str = "combined"


@router.post("/active-learning/score/{project_id}")
async def start_scoring(
    project_id: str,
    body: ScoreImagesRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Score all pending images by uncertainty — returns ranked list."""
    await get_owned_project(project_id, current_user, db)
    req = body or ScoreImagesRequest()
    task = score_unlabeled_images.delay(
        project_id, req.model_type, req.strategy, req.top_k, req.use_tta,
    )
    return {"task_id": task.id, "status": "queued"}


@router.post("/active-learning/curriculum-annotate/{project_id}")
async def start_curriculum_annotate(
    project_id: str,
    body: CurriculumAnnotateRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Smart auto-annotation with confidence tiers."""
    await get_owned_project(project_id, current_user, db)
    req = body or CurriculumAnnotateRequest()
    task = curriculum_auto_annotate.delay(
        project_id, req.high_conf, req.low_conf,
        req.review_band_top, req.review_band_bottom, req.use_tta,
    )
    return {"task_id": task.id, "status": "queued"}


@router.post("/active-learning/suggest/{project_id}")
async def start_suggest_review(
    project_id: str,
    body: SuggestReviewRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the top-N most uncertain images that need human annotation."""
    await get_owned_project(project_id, current_user, db)
    req = body or SuggestReviewRequest()
    task = suggest_for_review.delay(project_id, req.budget, req.strategy)
    return {"task_id": task.id, "status": "queued"}


# ── Dataset stats ────────────────────────────────────────────────

@router.get("/training-stats/{project_id}")
async def get_training_stats(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)

    all_imgs = await db.execute(select(Image).where(Image.project_id == project_id))
    images = all_imgs.scalars().all()

    total = len(images)
    annotated = [img for img in images if img.status == "annotated"]
    pending = [img for img in images if img.status == "pending"]

    annotated_ids = [img.id for img in annotated]
    class_counts = {}
    if annotated_ids:
        anns = await db.execute(
            select(Annotation.class_name, func.count(Annotation.id))
            .where(Annotation.image_id.in_(annotated_ids))
            .group_by(Annotation.class_name)
        )
        class_counts = {row[0]: row[1] for row in anns.fetchall()}

    total_anns = sum(class_counts.values())

    return {
        "total_images": total,
        "annotated_images": len(annotated),
        "pending_images": len(pending),
        "total_annotations": total_anns,
        "class_breakdown": class_counts,
        "ready_to_train": len(annotated) > 0,
    }


# ── Cancel task ───────────────────────────────────────────────────

@router.post("/cancel/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Revoke a Celery task and mark the DB job record as stopped."""
    # Verify the job belongs to the current user's project before cancelling
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == task_id))
    job = result.scalar_one_or_none()
    if job:
        proj_result = await db.execute(
            select(Project).where(
                Project.id == job.project_id,
                Project.user_id == current_user.id,
            )
        )
        if not proj_result.scalar_one_or_none():
            raise HTTPException(status_code=403, detail="Access denied")

    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")

    try:
        if job:
            job.status = "failure"
            job.result_meta = _sanitize_meta({
                **(job.result_meta or {}),
                "error": "Stopped by user",
            })
            job.finished_at = datetime.utcnow()
            await db.commit()
    except Exception:
        pass

    return {"task_id": task_id, "status": "revoked"}


# ── Task status ───────────────────────────────────────────────────

@router.get("/task-status/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return Celery task progress. Auth required; task_id is opaque so no ownership re-check needed."""
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.state,
        "result": None,
        "meta": None,
        "error": None,
    }
    if result.state == "SUCCESS":
        response["result"] = _sanitize_floats(result.result)
    elif result.state == "STARTED":
        try:
            response["meta"] = _sanitize_floats(result.info)
        except Exception:
            pass
    elif result.state == "FAILURE":
        response["error"] = str(result.result)
    return response


# ── Persistent job records ────────────────────────────────────────

def _sanitize_floats(obj):
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(item) for item in obj]
    return obj


def _sanitize_meta(obj):
    if isinstance(obj, str):
        return obj.encode("cp1252", errors="ignore").decode("cp1252")
    if isinstance(obj, dict):
        return {k: _sanitize_meta(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_meta(item) for item in obj]
    return obj


class JobCreateRequest(BaseModel):
    task_id: str
    project_id: str
    job_type: str = "seed_training"
    conf_used: Optional[float] = None
    result_meta: Optional[dict] = None


class JobUpdateRequest(BaseModel):
    status: Optional[str] = None
    result_meta: Optional[dict] = None
    finished_at: Optional[str] = None  # ISO-8601 string


@router.post("/jobs")
async def create_job(
    body: JobCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Persist a newly-submitted Celery job so it survives page reloads."""
    await get_owned_project(body.project_id, current_user, db)
    job = TrainingJob(
        id=body.task_id,
        project_id=body.project_id,
        job_type=body.job_type,
        status="pending",
        result_meta=_sanitize_meta(body.result_meta or {}),
        conf_used=body.conf_used,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return {"id": job.id, "created": True}


@router.get("/jobs/{project_id}")
async def list_jobs(
    project_id: str,
    job_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all persisted jobs for a project, oldest first."""
    await get_owned_project(project_id, current_user, db)

    q = select(TrainingJob).where(TrainingJob.project_id == project_id)
    if job_type:
        q = q.where(TrainingJob.job_type == job_type)
    q = q.order_by(TrainingJob.created_at.asc())

    result = await db.execute(q)
    jobs = result.scalars().all()

    return [
        {
            "id": job.id,
            "project_id": job.project_id,
            "job_type": job.job_type,
            "status": job.status,
            "result_meta": job.result_meta or {},
            "conf_used": job.conf_used,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        }
        for job in jobs
    ]


@router.patch("/jobs/{task_id}")
async def update_job(
    task_id: str,
    body: JobUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update status and/or result_meta for a job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == task_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {task_id} not found")

    # Verify job belongs to the current user's project
    proj_result = await db.execute(
        select(Project).where(
            Project.id == job.project_id,
            Project.user_id == current_user.id,
        )
    )
    if not proj_result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Access denied")

    if body.status is not None:
        job.status = body.status
    if body.result_meta is not None:
        job.result_meta = _sanitize_meta(body.result_meta)
    if body.finished_at is not None:
        job.finished_at = datetime.fromisoformat(body.finished_at).replace(tzinfo=None)

    await db.commit()
    return {"id": job.id, "status": job.status}
