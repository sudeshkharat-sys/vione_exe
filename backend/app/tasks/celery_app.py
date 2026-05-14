from celery import Celery
from ..config import settings

celery_app = Celery(
    "ai_vision",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.tasks.training",
        "app.tasks.auto_annotate",
        "app.tasks.ai_prompt",
        "app.tasks.video_processing",
        "app.tasks.active_learning",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    worker_pool="solo",  # Windows: avoids spawn-based prefork pool issues
)
