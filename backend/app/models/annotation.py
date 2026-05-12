import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..database import Base

class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id: Mapped[str] = mapped_column(String(36), ForeignKey("images.id"), nullable=False)
    class_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Normalized bbox [x_center, y_center, width, height]
    bbox: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    
    source: Mapped[str] = mapped_column(String(50), default="manual") # manual, auto
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    image = relationship("Image", back_populates="annotations")
