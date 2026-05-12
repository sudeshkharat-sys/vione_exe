import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column
from ..database import Base


class TrainingJob(Base):
    """
    Persisted record of every Celery training / auto-annotate job.

    The primary key is the Celery task ID so we can look up status
    from the Celery backend without a separate ID.

    The `seq` column (BigInteger with a Sequence) lives in the Core
    table definition (table_creation.py) and is auto-filled by
    PostgreSQL — we deliberately omit it here so ORM INSERTs don't
    need to supply it.
    """
    __tablename__ = "training_jobs"

    # Celery task ID is the natural PK
    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    job_type: Mapped[str] = mapped_column(
        String(50), default="seed_training"
    )  # seed_training | auto_annotate

    status: Mapped[str] = mapped_column(
        String(50), default="pending"
    )  # pending | started | success | failure

    # Full snapshot of frontend state (logs, charts, result, …)
    result_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Confidence threshold — only meaningful for auto_annotate jobs
    conf_used: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
