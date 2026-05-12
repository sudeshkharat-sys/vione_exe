import uuid
from datetime import datetime
from sqlalchemy import String, Integer, Float, BigInteger, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..database import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    filepath: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    duration: Mapped[float] = mapped_column(Float, nullable=True)   # seconds
    fps: Mapped[float] = mapped_column(Float, nullable=True)
    width: Mapped[int] = mapped_column(Integer, nullable=True)
    height: Mapped[int] = mapped_column(Integer, nullable=True)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=True)
    # uploaded | extracting | stopped | done | failed
    status: Mapped[str] = mapped_column(String(50), default="uploaded")
    frames_extracted: Mapped[int] = mapped_column(Integer, default=0)
    task_id: Mapped[str] = mapped_column(String(36), nullable=True)  # Celery task ID
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="videos")
