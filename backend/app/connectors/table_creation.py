"""
table_creation.py
~~~~~~~~~~~~~~~~~
SQLAlchemy **Core** table definitions for the AI Vision Platform.

Why Core instead of ORM declarative?
--------------------------------------
* Gives exact control over DDL (column types, indexes, constraints).
* JSONB is a PostgreSQL-native type that stores JSON as binary — it supports
  GIN indexing and `@>` / `->` operators unavailable with generic ``JSON``.
* The ``DYNAMIC_TABLES`` registry lets other modules look up Table objects
  without importing circular ORM models.

server_default note
-------------------
Always pass SQL expressions to ``server_default`` as ``text(...)`` objects.
A plain Python string gets double-escaped by SQLAlchemy's DDL compiler,
turning ``'pending'`` into ``'''pending'''`` — invalid SQL.  ``text()``
marks the value as a raw SQL fragment and prevents the extra quoting.

Adding a new table
------------------
1. Define it with :func:`create_dynamic_table`.
2. It is automatically registered in :data:`DYNAMIC_TABLES`.
3. Call ``metadata.create_all(engine)`` or use :class:`StateDBManager`.
"""

from __future__ import annotations

import uuid
from typing import Dict

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Sequence,
    String,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB

# ── Shared metadata ───────────────────────────────────────────────────────────
# All tables live in the same ``MetaData`` object so ``metadata.create_all``
# creates them in one shot and foreign-key resolution works correctly.
metadata = MetaData()

# Runtime registry: table_name → Table
DYNAMIC_TABLES: Dict[str, Table] = {}


def create_dynamic_table(name: str, *columns: Column, **kw) -> Table:
    """
    Create a :class:`~sqlalchemy.Table`, bind it to :data:`metadata`, and
    register it in :data:`DYNAMIC_TABLES`.

    Extra keyword arguments (``extend_existing``, ``schema``, …) are forwarded
    to the ``Table`` constructor.

    Returns the new :class:`~sqlalchemy.Table` instance.
    """
    table = Table(name, metadata, *columns, **kw)
    DYNAMIC_TABLES[name] = table
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Table: projects
# ─────────────────────────────────────────────────────────────────────────────
projects_table = create_dynamic_table(
    "projects",
    Column(
        "id",
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="UUID primary key",
    ),
    Column("name", String(255), nullable=False, comment="Human-readable project name"),
    Column("description", Text, nullable=True, comment="Optional longer description"),
    # JSONB: list of class label strings e.g. ["cat", "dog"]
    # text() prevents SQLAlchemy from double-quoting the expression in DDL
    Column(
        "classes",
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment="Ordered list of annotation class labels",
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Row creation timestamp (UTC)",
    ),
    Column(
        "user_id",
        String(36),
        nullable=True,
        index=True,
        comment="Owner user ID (FK to users.id)",
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Table: images
# ─────────────────────────────────────────────────────────────────────────────
images_table = create_dynamic_table(
    "images",
    Column("id", String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column(
        "project_id",
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("filename", String(512), nullable=False, comment="Original file name"),
    Column("filepath", String(1024), nullable=False, comment="Server-relative URL path"),
    Column("width", Integer, nullable=False),
    Column("height", Integer, nullable=False),
    Column(
        "status",
        String(50),
        nullable=False,
        server_default=text("'pending'"),
        comment="pending | annotated",
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    # Composite index: most queries filter by (project_id, status)
    Index("ix_images_project_status", "project_id", "status"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Table: annotations
# ─────────────────────────────────────────────────────────────────────────────
annotations_table = create_dynamic_table(
    "annotations",
    Column("id", String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column(
        "image_id",
        String(36),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("class_name", String(255), nullable=False),
    # JSONB: [x_center, y_center, width, height] normalised 0-1
    Column(
        "bbox",
        JSONB,
        nullable=True,
        comment="Normalised bounding box [x_c, y_c, w, h]",
    ),
    Column(
        "source",
        String(50),
        nullable=False,
        server_default=text("'manual'"),
        comment="manual | auto",
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    # GIN index on bbox JSONB for spatial/containment queries
    Index(
        "ix_annotations_bbox_gin",
        "bbox",
        postgresql_using="gin",
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Table: training_jobs
# Persists Celery training task metadata so the UI can list historic jobs.
# ─────────────────────────────────────────────────────────────────────────────

# Auto-increment surrogate key (useful for ordering without UUID comparison)
_training_jobs_seq = Sequence("training_jobs_seq", metadata=metadata)

training_jobs_table = create_dynamic_table(
    "training_jobs",
    Column(
        "id",
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Celery task ID stored as PK for easy lookup",
    ),
    Column(
        "seq",
        BigInteger,
        _training_jobs_seq,
        server_default=_training_jobs_seq.next_value(),
        nullable=False,
        unique=True,
        comment="Monotonic ordering key",
    ),
    Column(
        "project_id",
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column(
        "job_type",
        String(50),
        nullable=False,
        server_default=text("'seed_training'"),
        comment="seed_training | auto_annotate | …",
    ),
    Column(
        "status",
        String(50),
        nullable=False,
        server_default=text("'queued'"),
        comment="queued | started | success | failure",
    ),
    # JSONB blob: stores epoch history, loss curves, mAP, etc.
    Column(
        "result_meta",
        JSONB,
        nullable=True,
        comment="Arbitrary result / progress payload from Celery",
    ),
    Column(
        "conf_used",
        Float,
        nullable=True,
        comment="Confidence threshold (auto-annotate jobs only)",
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
    Column(
        "finished_at",
        DateTime(timezone=True),
        nullable=True,
    ),
    Index("ix_training_jobs_project_status", "project_id", "status"),
)

# ─────────────────────────────────────────────────────────────────────────────
# Table: videos
# Stores uploaded video files and frame-extraction metadata.
# Extracted frames are stored as regular Image rows linked by video_id.
# ─────────────────────────────────────────────────────────────────────────────
videos_table = create_dynamic_table(
    "videos",
    Column("id", String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
    Column(
        "project_id",
        String(36),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("original_filename", String(512), nullable=False),
    Column("filepath", String(1024), nullable=False),
    Column("file_size", BigInteger, nullable=False, server_default=text("0")),
    Column("duration", Float, nullable=True, comment="Video duration in seconds"),
    Column("fps", Float, nullable=True),
    Column("width", Integer, nullable=True),
    Column("height", Integer, nullable=True),
    Column("total_frames", Integer, nullable=True),
    Column(
        "status",
        String(50),
        nullable=False,
        server_default=text("'uploaded'"),
        comment="uploaded | extracting | done | failed",
    ),
    Column("frames_extracted", Integer, nullable=False, server_default=text("0")),
    Column("task_id", String(36), nullable=True, comment="Celery task ID while extracting"),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    ),
)
