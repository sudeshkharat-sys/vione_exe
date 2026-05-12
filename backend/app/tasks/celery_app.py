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
    task_acks_late=True,              # re-queue task if worker crashes mid-run
    task_reject_on_worker_lost=True,  # don't drop tasks on hard worker crash
    task_soft_time_limit=3600,        # 1h soft limit → SoftTimeLimitExceeded
    task_time_limit=3900,             # 65min hard kill (15min grace after soft)
    worker_max_tasks_per_child=50,    # recycle worker to prevent memory leaks
    # ── Worker heartbeat / events ─────────────────────────────────────────────
    # These allow inspect().active() to detect a worker that is busy with
    # GPU training and therefore cannot respond to ping() in time.
    # Without these, the UI incorrectly reports "No Celery worker found"
    # while training is visibly running in the terminal.
    worker_send_task_events=True,   # worker proactively publishes task events
    task_send_sent_event=True,      # event fired when a task is dispatched
    worker_heartbeat=10,            # publish heartbeat every 10 s
)
