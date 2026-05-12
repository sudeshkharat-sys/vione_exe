from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict
from datetime import datetime


# ── Auth schemas ───────────────────────────────────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str = Field(..., min_length=8, max_length=128)

class UserLogin(BaseModel):
    email: str
    password: str = Field(..., min_length=8, max_length=128)

class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    token: str
    user: UserResponse

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    classes: List[str]

class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    classes: Optional[List[str]] = None

class ClassRenameRequest(BaseModel):
    old_name: str
    new_name: str

class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    classes: List[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ImageResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    filepath: str
    width: int
    height: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class AnnotationCreate(BaseModel):
    image_id: str
    class_name: str
    bbox: Optional[List[float]] = None
    source: str = "manual"

class AnnotationResponse(BaseModel):
    id: str
    image_id: str
    class_name: str
    bbox: Optional[List[float]]
    source: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Video schemas ───────────────────────────────────────────────────
class VideoResponse(BaseModel):
    id: str
    project_id: str
    original_filename: str
    filepath: str
    file_size: int
    duration: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    total_frames: Optional[int]
    status: str
    frames_extracted: int
    task_id: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class VideoFrameExtractionRequest(BaseModel):
    # Extract 1 frame every `sample_every_n` frames (e.g. 30 = 1 fps at 30fps video)
    sample_every_n: int = Field(default=30, ge=1, le=3000)
    # Hard cap on total frames extracted (0 = no limit)
    max_frames: int = Field(default=300, ge=0, le=10000)
