from pathlib import Path
from functools import lru_cache
from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # ── File system ───────────────────────────────────────────────────────────
    upload_dir: Path = Path("./data/uploads")
    model_dir: Path = Path("./data/models")
    # Pre-downloaded YOLO base weights for offline deployment.
    # Populated during Docker build by scripts/download_yolo_weights.py.
    # Not volume-mounted so weights survive container restarts without internet.
    yolo_weights_dir: Path = Path("/app/data/yolo_weights")

    # ── Optional HF model paths ───────────────────────────────────────────────
    grounding_dino_path: str = "../datavision_hf_models/grounding-dino-base"
    sam2_path: str = "../datavision_hf_models/sam2-hiera-large"
    siglip_path: str = "../datavision_hf_models/siglip-so400m-patch14-384"

    # ── Training ──────────────────────────────────────────────────────────────
    # Raised from 0.0005 → 0.005: the original value was 20× below YOLO's
    # default (0.01), which starved learning on small inspection datasets.
    # 0.005 with cosine decay (lrf=0.01) gives a healthy learning curve
    # while staying conservative enough not to blow up on few images.
    seed_learning_rate: float = 0.005
    # Main model LR; when fine-tuning from seed weights the task halves this
    # automatically to avoid catastrophic forgetting / hallucination.
    main_learning_rate: float = 0.005

    # ── Auto-annotation defaults ───────────────────────────────────────────
    # Minimum confidence for auto-annotations (0.25 balances recall vs
    # hallucination; the old default of 0.1 produced ~60 % false positives).
    auto_annotate_conf: float = 0.25
    # DINO zero-shot detection threshold (raised from 0.15 to reduce noise).
    dino_box_threshold: float = 0.25
    dino_text_threshold: float = 0.25

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_vision"
    postgres_user: str = "postgres"
    postgres_password: str = "password"

    # ── SQLAlchemy pool (sync connector) ─────────────────────────────────────
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_pool_recycle: int = 1800   # 30 min — keeps connections fresh

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Ignore stale keys in .env (e.g. the old DATABASE_URL from SQLite)
        "extra": "ignore",
        # Suppress the false-positive "model_dir conflicts with model_" warning
        "protected_namespaces": ("settings_",),
    }

    # ── Computed connection strings ───────────────────────────────────────────
    @computed_field  # type: ignore[misc]
    @property
    def postgres_url(self) -> str:
        """Async DSN for SQLAlchemy (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def postgres_url_sync(self) -> str:
        """Sync DSN for SQLAlchemy (psycopg2 driver — used by connectors)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[misc]
    @property
    def postgres_url_raw(self) -> str:
        """Plain psycopg2 DSN (no dialect prefix) — used for CREATE DATABASE."""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname=postgres user={self.postgres_user} password={self.postgres_password}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — reads .env only once per process."""
    return Settings()


# Convenience alias kept for backward-compat with existing imports
settings = get_settings()
