from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from ..database import get_db
from ..models.annotation import Annotation
from ..models.image import Image
from ..models.project import Project
from ..models.user import User
from ..api.auth import get_current_user
from ..api.deps import get_owned_project

router = APIRouter(prefix="/annotations", tags=["annotations"])


class AnnotationCreate(BaseModel):
    image_id: str
    class_name: str
    bbox: List[float]
    source: Optional[str] = "manual"


class AnnotationUpdate(BaseModel):
    class_name: Optional[str] = None
    bbox: Optional[List[float]] = None
    source: Optional[str] = None


class BulkAnnotationCreate(BaseModel):
    annotations: List[AnnotationCreate]
    clear_existing: bool = False


@router.post("")
async def create_annotation(
    annotation: AnnotationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Image).where(Image.id == annotation.image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    await get_owned_project(image.project_id, current_user, db)

    new_annotation = Annotation(
        image_id=annotation.image_id,
        class_name=annotation.class_name,
        bbox=annotation.bbox,
        source=annotation.source or "manual",
    )
    db.add(new_annotation)

    image.status = "annotated"
    await db.commit()
    await db.refresh(new_annotation)

    return {
        "id": new_annotation.id,
        "image_id": new_annotation.image_id,
        "class_name": new_annotation.class_name,
        "bbox": new_annotation.bbox,
        "source": new_annotation.source,
    }


@router.get("/image/{image_id}")
async def get_annotations(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    await get_owned_project(image.project_id, current_user, db)

    result = await db.execute(
        select(Annotation).where(Annotation.image_id == image_id)
    )
    annotations = result.scalars().all()

    return [
        {
            "id": ann.id,
            "image_id": ann.image_id,
            "class_name": ann.class_name,
            "bbox": ann.bbox,
            "source": ann.source,
        }
        for ann in annotations
    ]


@router.put("/{annotation_id}")
async def update_annotation(
    annotation_id: str,
    update: AnnotationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Annotation).where(Annotation.id == annotation_id)
    )
    annotation = result.scalar_one_or_none()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    result = await db.execute(select(Image).where(Image.id == annotation.image_id))
    image = result.scalar_one_or_none()
    await get_owned_project(image.project_id, current_user, db)

    if update.class_name is not None:
        annotation.class_name = update.class_name
    if update.bbox is not None:
        annotation.bbox = update.bbox
    if update.source is not None:
        annotation.source = update.source

    await db.commit()
    await db.refresh(annotation)

    return {
        "id": annotation.id,
        "image_id": annotation.image_id,
        "class_name": annotation.class_name,
        "bbox": annotation.bbox,
        "source": annotation.source,
    }


@router.delete("/{annotation_id}")
async def delete_annotation(
    annotation_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Annotation).where(Annotation.id == annotation_id)
    )
    annotation = result.scalar_one_or_none()
    if not annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")

    result = await db.execute(select(Image).where(Image.id == annotation.image_id))
    image = result.scalar_one_or_none()
    await get_owned_project(image.project_id, current_user, db)

    await db.delete(annotation)

    remaining = await db.execute(
        select(Annotation).where(Annotation.image_id == annotation.image_id)
    )
    if not remaining.scalars().all():
        image.status = "pending"

    await db.commit()
    return {"status": "deleted"}


@router.post("/bulk")
async def bulk_create_annotations(
    body: BulkAnnotationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not body.annotations:
        return {"created": 0}

    image_ids = list({a.image_id for a in body.annotations})

    for image_id in image_ids:
        result = await db.execute(select(Image).where(Image.id == image_id))
        image = result.scalar_one_or_none()
        if not image:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        await get_owned_project(image.project_id, current_user, db)

    if body.clear_existing:
        await db.execute(
            delete(Annotation).where(Annotation.image_id.in_(image_ids))
        )

    new_anns = []
    for a in body.annotations:
        ann = Annotation(
            image_id=a.image_id,
            class_name=a.class_name,
            bbox=a.bbox,
            source=a.source or "manual",
        )
        db.add(ann)
        new_anns.append(ann)

    for image_id in image_ids:
        result = await db.execute(select(Image).where(Image.id == image_id))
        image = result.scalar_one_or_none()
        if image:
            image.status = "annotated"

    await db.commit()
    return {"created": len(new_anns)}
