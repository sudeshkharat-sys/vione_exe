from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from ..database import get_db
from ..models.project import Project
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User
from ..schemas.base import ProjectCreate, ProjectResponse, ProjectUpdateRequest, ClassRenameRequest
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from typing import List, Dict
import shutil

router = APIRouter(prefix="/projects", tags=["projects"])




@router.post("", response_model=ProjectResponse)
async def create_project(
    data: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = Project(
        name=data.name,
        description=data.description,
        classes=data.classes,
        user_id=current_user.id,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)
    return project


@router.get("", response_model=List[ProjectResponse])
async def list_projects(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Project).where(Project.user_id == current_user.id))
    return result.scalars().all()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await get_owned_project(project_id, current_user, db)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    data: ProjectUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update project name, description, or classes list."""
    project = await get_owned_project(project_id, current_user, db)

    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.classes is not None:
        project.classes = data.classes

    await db.commit()
    await db.refresh(project)
    return project


@router.patch("/{project_id}/rename-class")
async def rename_class(
    project_id: str,
    data: ClassRenameRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename a class in project.classes AND in all existing annotations."""
    project = await get_owned_project(project_id, current_user, db)

    # Replace in the classes list
    updated_classes = [
        data.new_name if c == data.old_name else c
        for c in project.classes
    ]
    project.classes = updated_classes

    # Bulk-update annotation class_name via image subquery
    image_ids_subq = (
        select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    )
    await db.execute(
        update(Annotation)
        .where(
            Annotation.image_id.in_(image_ids_subq),
            Annotation.class_name == data.old_name,
        )
        .values(class_name=data.new_name)
    )

    await db.commit()
    return {
        "updated_classes": updated_classes,
        "old_name": data.old_name,
        "new_name": data.new_name,
    }


@router.delete("/{project_id}/class-annotations/{class_name}")
async def delete_class_annotations(
    project_id: str,
    class_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete all annotations that use a specific class_name in a project."""
    await get_owned_project(project_id, current_user, db)
    image_ids_subq = (
        select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    )
    result = await db.execute(
        delete(Annotation)
        .where(
            Annotation.image_id.in_(image_ids_subq),
            Annotation.class_name == class_name,
        )
        .returning(Annotation.id)
    )
    deleted_count = len(result.fetchall())
    await db.commit()
    return {"deleted_count": deleted_count, "class_name": class_name}


@router.get("/{project_id}/class-stats")
async def get_class_stats(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, int]:
    """Return annotation count per class_name for a project."""
    await get_owned_project(project_id, current_user, db)
    image_ids_subq = (
        select(Image.id).where(Image.project_id == project_id).scalar_subquery()
    )
    result = await db.execute(
        select(Annotation.class_name, func.count(Annotation.id).label("cnt"))
        .where(Annotation.image_id.in_(image_ids_subq))
        .group_by(Annotation.class_name)
    )
    rows = result.fetchall()
    return {row.class_name: row.cnt for row in rows}


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently delete a project and ALL of its data:
      - Annotations (via image cascade)
      - Images (SQLAlchemy cascade on project → images → annotations)
      - Training jobs (DB-level FK ON DELETE CASCADE)
      - Uploaded files on disk
    """
    project = await get_owned_project(project_id, current_user, db)

    # ORM delete — cascades to images → annotations automatically
    await db.delete(project)
    await db.commit()

    # Remove uploaded image files from disk
    project_upload_dir = settings.upload_dir / project_id
    if project_upload_dir.exists():
        shutil.rmtree(project_upload_dir, ignore_errors=True)

    return {"message": "Project deleted successfully", "id": project_id}
