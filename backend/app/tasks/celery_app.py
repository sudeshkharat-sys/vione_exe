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
    task_acks_late=False,             # ack immediately — prevents zombie re-runs on OOM crash
    task_reject_on_worker_lost=False, # don't re-queue on crash (training should be explicit)
    task_soft_time_limit=3600,        # 1h soft limit → SoftTimeLimitExceeded
    task_time_limit=3900,             # 65min hard kill (15min grace after soft)
    worker_max_tasks_per_child=50,    # recycle worker to prevent memory leaks
)
