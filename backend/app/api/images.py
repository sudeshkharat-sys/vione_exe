from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import uuid, os, shutil
from ..database import get_db
from ..models.image import Image
from ..models.annotation import Annotation
from ..models.project import Project
from ..models.user import User
from ..api.auth import get_current_user
from ..api.deps import get_owned_project
from ..config import settings
from PIL import Image as PILImage

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload/{project_id}")
async def upload_image(
    project_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)

    image_id = str(uuid.uuid4())
    project_upload_dir = settings.upload_dir / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)

    file_extension = os.path.splitext(file.filename)[1].lower()
    filename = f"{image_id}{file_extension}"
    file_path = project_upload_dir / filename

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    try:
        with PILImage.open(file_path) as pil_img:
            width, height = pil_img.size
    except Exception:
        width, height = 0, 0

    relative_path = f"/data/uploads/{project_id}/{filename}"

    new_image = Image(
        id=image_id,
        project_id=project_id,
        filename=file.filename,
        filepath=relative_path,
        width=width,
        height=height,
        status="pending",
    )
    db.add(new_image)
    await db.commit()
    await db.refresh(new_image)

    return {
        "id": new_image.id,
        "filename": new_image.filename,
        "filepath": new_image.filepath,
        "width": new_image.width,
        "height": new_image.height,
        "status": new_image.status,
    }


@router.get("/{project_id}")
async def get_images(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)

    result = await db.execute(
        select(Image).where(Image.project_id == project_id)
    )
    images = result.scalars().all()

    image_list = []
    for img in images:
        ann_count_result = await db.execute(
            select(func.count(Annotation.id)).where(Annotation.image_id == img.id)
        )
        ann_count = ann_count_result.scalar()
        image_list.append({
            "id": img.id,
            "filename": img.filename,
            "filepath": img.filepath,
            "width": img.width,
            "height": img.height,
            "status": img.status,
            "annotation_count": ann_count,
        })

    return image_list


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    await get_owned_project(image.project_id, current_user, db)

    file_path = settings.upload_dir.parent / image.filepath.lstrip("/")
    if file_path.exists():
        file_path.unlink()

    await db.delete(image)
    await db.commit()
    return {"status": "deleted"}


@router.patch("/{image_id}/status")
async def update_image_status(
    image_id: str,
    status: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    await get_owned_project(image.project_id, current_user, db)

    image.status = status
    await db.commit()
    return {"id": image_id, "status": status}


@router.post("/bulk-upload/{project_id}")
async def bulk_upload_images(
    project_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)

    project_upload_dir = settings.upload_dir / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    for file in files:
        image_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        filename = f"{image_id}{file_extension}"
        file_path = project_upload_dir / filename

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        try:
            with PILImage.open(file_path) as pil_img:
                width, height = pil_img.size
        except Exception:
            width, height = 0, 0

        relative_path = f"/data/uploads/{project_id}/{filename}"
        new_image = Image(
            id=image_id,
            project_id=project_id,
            filename=file.filename,
            filepath=relative_path,
            width=width,
            height=height,
            status="pending",
        )
        db.add(new_image)
        uploaded.append({"id": image_id, "filename": file.filename})

    await db.commit()
    return {"uploaded": len(uploaded), "images": uploaded}
