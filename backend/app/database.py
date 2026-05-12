"""
database.py
~~~~~~~~~~~
Async SQLAlchemy engine + session factory for FastAPI route handlers.

Uses asyncpg as the async driver for PostgreSQL.  The ORM ``Base`` is kept
here so existing model imports (``from app.database import Base``) continue
to work unchanged.

On startup the application calls :func:`init_db` which delegates to
:class:`~app.connectors.statedb_manager.StateDBManager` so the database and
all tables are created if they do not exist yet.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

_settings = get_settings()

# ── Async engine (asyncpg) ────────────────────────────────────────────────────
engine = create_async_engine(
    _settings.postgres_url,   # postgresql+asyncpg://…
    echo=False,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_timeout=_settings.db_pool_timeout,
    pool_recycle=_settings.db_pool_recycle,
    pool_pre_ping=True,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncSession:  # type: ignore[return]
    """
    Yield an :class:`AsyncSession` for use as a FastAPI dependency.

    Commits on clean exit, rolls back on exception, and always closes the
    session when the request finishes.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Application startup ───────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Ensure the target PostgreSQL database and all application tables exist.

    Called from ``app.main`` lifespan handler on startup.  Uses the
    synchronous :class:`~app.connectors.statedb_manager.StateDBManager` (runs
    in a thread executor to avoid blocking the event loop).

    The ORM ``Base.metadata.create_all`` is also called on the async engine so
    any model-defined tables that are not in ``table_creation.py`` are still
    created.
    """
    import asyncio

    # 1. Bootstrap via StateDBManager (creates DB + Core tables)
    def _sync_bootstrap() -> None:
        from .connectors.statedb_manager import StateDBManager
        StateDBManager().initialize_database()

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_bootstrap)

    # 2. Ensure ORM-declared models are also reflected (belt + suspenders)
    from .models.project import Project           # noqa: F401
    from .models.image import Image               # noqa: F401
    from .models.annotation import Annotation     # noqa: F401
    from .models.training_job import TrainingJob  # noqa: F401
    from .models.user import User                 # noqa: F401
    from .models.video import Video               # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Migration: add user_id to projects for existing installations
        from sqlalchemy import text
        await conn.execute(text(
            "ALTER TABLE projects ADD COLUMN IF NOT EXISTS user_id VARCHAR(36)"
        ))
