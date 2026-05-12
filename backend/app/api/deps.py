"""
deps.py
~~~~~~~
Shared FastAPI dependency helpers for resource ownership verification.

Every helper fetches a resource by ID and verifies it belongs to the
current user (directly or via the project → user_id chain).  A 404 is
returned when the record does not exist; a 403 when it exists but is
owned by a different user.  This ensures no authenticated user can read
or modify another user's data.
"""

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.project import Project
from ..models.image import Image
from ..models.video import Video
from ..models.annotation import Annotation
from ..models.user import User


async def get_owned_project(project_id: str, current_user: User, db: AsyncSession) -> Project:
    """Return the project if it belongs to current_user, else raise 404/403."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if project.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    return project


async def get_owned_image(image_id: str, current_user: User, db: AsyncSession) -> Image:
    """Return the image if its project belongs to current_user, else raise 404/403."""
    result = await db.execute(
        select(Image)
        .join(Project, Image.project_id == Project.id)
        .where(Image.id == image_id, Project.user_id == current_user.id)
    )
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image


async def get_owned_video(video_id: str, current_user: User, db: AsyncSession) -> Video:
    """Return the video if its project belongs to current_user, else raise 404/403."""
    result = await db.execute(
        select(Video)
        .join(Project, Video.project_id == Project.id)
        .where(Video.id == video_id, Project.user_id == current_user.id)
    )
    video = result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


async def get_owned_annotation(annotation_id: str, current_user: User, db: AsyncSession) -> Annotation:
    """Return the annotation if its image's project belongs to current_user, else raise 404/403."""
    result = await db.execute(
        select(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .join(Project, Image.project_id == Project.id)
        .where(Annotation.id == annotation_id, Project.user_id == current_user.id)
    )
    ann = result.scalar_one_or_none()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return ann
