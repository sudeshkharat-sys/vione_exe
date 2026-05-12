from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models.image import Image
from ..models.user import User
from ..schemas.base import ImageResponse
from ..config import settings
from ..api.auth import get_current_user
from ..api.deps import get_owned_project, get_owned_image
from typing import List
import shutil
import os
import uuid
from PIL import Image as PILImage

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload/{project_id}", response_model=List[ImageResponse])
async def upload_images(
    project_id: str,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)

    project_dir = settings.upload_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    uploaded_images = []
    for file in files:
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = project_dir / unique_filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with PILImage.open(file_path) as img:
            width, height = img.size
            # Correct dimensions for EXIF orientation so stored values match
            # what the browser actually displays (phones often shoot rotated).
            # Orientations 5-8 require a 90°/270° rotation, swapping W and H.
            try:
                exif_data = img._getexif()
                if exif_data:
                    orientation = exif_data.get(274, 1)  # tag 274 = Orientation
                    if orientation in (5, 6, 7, 8):
                        width, height = height, width
            except Exception:
                pass

        db_image = Image(
            project_id=project_id,
            filename=file.filename,
            filepath=f"/uploads/{project_id}/{unique_filename}",
            width=width,
            height=height,
        )
        db.add(db_image)
        uploaded_images.append(db_image)

    await db.commit()
    for img in uploaded_images:
        await db.refresh(img)

    return uploaded_images


@router.get("/project/{project_id}", response_model=List[ImageResponse])
async def list_project_images(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_project(project_id, current_user, db)
    result = await db.execute(select(Image).where(Image.project_id == project_id))
    return result.scalars().all()


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    image = await get_owned_image(image_id, current_user, db)

    # Remove the file from disk (same two-candidate pattern as video deletion)
    rel = image.filepath.lstrip("/")
    for candidate in [
        settings.upload_dir.resolve().parent / rel,
        settings.upload_dir.resolve() / rel,
    ]:
        try:
            if candidate.exists():
                candidate.unlink()
                break
        except Exception:
            pass

    await db.delete(image)
    await db.commit()


@router.patch("/{image_id}/mark-empty", response_model=ImageResponse)
async def mark_image_empty(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an image as annotated with no objects (negative/background frame).
    The image status is set to 'annotated' with zero annotation rows.
    YOLO training will generate an empty label file for it, which is valid
    and teaches the model to suppress false positives on blank frames.
    """
    image = await get_owned_image(image_id, current_user, db)
    image.status = "annotated"
    await db.commit()
    await db.refresh(image)
    return image
