"""
videos.py
~~~~~~~~~
API endpoints for video upload and frame extraction.

Workflow
--------
1. POST /videos/upload/{project_id}   — save video file, insert Video row
2. GET  /videos/project/{project_id}  — list all videos in a project
3. GET  /videos/{video_id}            — single video status/metadata
4. POST /videos/{video_id}/extract-frames — kick off the Celery frame-extraction task
5. POST /videos/{video_id}/stop-extraction — cancel a running extraction
6. DELETE /videos/{video_id}          — remove video + extracted frame Image rows
"""

import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from ..database import get_db
from ..models.video import Video
from ..models.image import Image
from ..models.user import User
from ..schemas.base import VideoResponse, VideoFrameExtractionRequest
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_video

router = APIRouter(prefix="/videos", tags=["videos"])

# Allowed video MIME types / extensions
_ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv"}


@router.post("/upload/{project_id}", response_model=VideoResponse)
async def upload_video(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a single video file and create a Video record."""
    await get_owned_project(project_id, current_user, db)

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Allowed: {sorted(_ALLOWED_VIDEO_EXTS)}",
        )

    video_dir = settings.upload_dir / project_id / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = video_dir / unique_filename

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    file_size = file_path.stat().st_size

    db_video = Video(
        project_id=project_id,
        original_filename=file.filename or unique_filename,
        filepath=f"/uploads/{project_id}/videos/{unique_filename}",
        file_size=file_size,
        status="uploaded",
    )
    db.add(db_video)
    await db.commit()
    await db.refresh(db_video)
    return db_video


@router.get("/project/{project_id}", response_model=List[VideoResponse])
async def list_project_videos(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return all videos belonging to a project."""
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(
        select(Video)
        .where(Video.project_id == project_id)
        .order_by(Video.created_at.desc())
    )
    return result.scalars().all()


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Fetch a single video by ID (useful for polling extraction status)."""
    return await get_owned_video(video_id, current_user, db)


@router.post("/{video_id}/extract-frames", response_model=VideoResponse)
async def extract_frames(
    video_id: str,
    body: VideoFrameExtractionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Kick off a Celery task to extract frames from the video.
    Returns the video record immediately (status will be 'extracting').
    The frontend should poll GET /videos/{video_id} to watch progress.
    """
    video = await get_owned_video(video_id, current_user, db)

    if video.status == "extracting":
        raise HTTPException(status_code=409, detail="Extraction already in progress")

    from ..tasks.video_processing import extract_video_frames

    task = extract_video_frames.delay(
        video_id,
        sample_every_n=body.sample_every_n,
        max_frames=body.max_frames,
    )

    video.status = "extracting"
    video.frames_extracted = 0
    video.task_id = task.id
    await db.commit()
    await db.refresh(video)
    return video


@router.post("/{video_id}/stop-extraction", response_model=VideoResponse)
async def stop_extraction(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running frame-extraction task."""
    video = await get_owned_video(video_id, current_user, db)

    if video.status != "extracting":
        raise HTTPException(status_code=409, detail="No extraction in progress")

    if video.task_id:
        from ..tasks.celery_app import celery_app
        celery_app.control.revoke(video.task_id, terminate=True, signal="SIGTERM")

    video.status = "stopped"
    video.task_id = None
    await db.commit()
    await db.refresh(video)
    return video


@router.delete("/{video_id}", status_code=204)
async def delete_video(
    video_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a video and all Image rows that were extracted from it."""
    video = await get_owned_video(video_id, current_user, db)

    # Delete extracted frame Image rows (identified by filepath pattern)
    frame_path_prefix = f"/uploads/{video.project_id}/video_frames/{video_id}/"
    await db.execute(
        delete(Image).where(Image.filepath.like(f"{frame_path_prefix}%"))
    )

    # Remove the video file from disk
    rel = video.filepath.lstrip("/")
    for anchor in [
        settings.upload_dir.resolve().parent / rel,
        settings.upload_dir.resolve() / rel,
    ]:
        try:
            if anchor.exists():
                anchor.unlink()
                break
        except Exception:
            pass

    # Remove extracted frames directory
    frames_dir = settings.upload_dir / video.project_id / "video_frames" / video_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)

    await db.delete(video)
    await db.commit()
