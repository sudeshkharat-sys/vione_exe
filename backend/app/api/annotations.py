from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.user import User
from ..schemas.base import AnnotationCreate, AnnotationResponse
from ..api.auth import get_current_user
from ..api.deps import get_owned_image, get_owned_annotation
from typing import List
from pydantic import BaseModel


class ClassifyBody(BaseModel):
    class_name: str


router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.post("", response_model=AnnotationResponse)
async def create_annotation(
    data: AnnotationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify the target image belongs to the current user
    image = await get_owned_image(data.image_id, current_user, db)

    annotation = Annotation(
        image_id=data.image_id,
        class_name=data.class_name,
        bbox=data.bbox,
        source=data.source,
    )
    db.add(annotation)
    image.status = "annotated"
    await db.commit()
    await db.refresh(annotation)
    return annotation


@router.get("/image/{image_id}", response_model=List[AnnotationResponse])
async def list_image_annotations(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await get_owned_image(image_id, current_user, db)
    result = await db.execute(select(Annotation).where(Annotation.image_id == image_id))
    return result.scalars().all()


@router.patch("/{annotation_id}/classify", response_model=AnnotationResponse)
async def classify_annotation(
    annotation_id: str,
    data: ClassifyBody,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Assign a class name to an AI-detected annotation and promote it to manual."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    ann.class_name = data.class_name
    ann.source = "manual"
    await db.commit()
    await db.refresh(ann)
    return ann


@router.patch("/{annotation_id}/verify")
async def verify_annotation(
    annotation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark an AI annotation as verified by changing source to 'manual'."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    ann.source = "manual"
    await db.commit()
    return {"status": "verified", "id": annotation_id}


@router.delete("/{annotation_id}")
async def delete_annotation(
    annotation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an annotation (used for 'Reject')."""
    ann = await get_owned_annotation(annotation_id, current_user, db)
    image_id = ann.image_id
    await db.delete(ann)

    # If no annotations remain, set image status back to pending
    count_res = await db.execute(
        select(Annotation).where(Annotation.image_id == image_id)
    )
    if not count_res.scalars().first():
        img_res = await db.execute(select(Image).where(Image.id == image_id))
        img = img_res.scalar_one_or_none()
        if img:
            img.status = "pending"

    await db.commit()
    return {"status": "deleted"}
