from .celery_app import celery_app
from ultralytics import YOLO
from pathlib import Path
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
import uuid
import json
import os
import numpy as np


def _resolve_image_path(filepath: str) -> Path | None:
    """
    Try several strategies to locate an uploaded image file.

    The Celery worker may run from a different CWD than the FastAPI server,
    so Path(".") / filepath is unreliable.  We resolve to absolute paths
    anchored to settings.upload_dir instead.
    """
    # Strip any leading slash so we can join safely
    rel = filepath.lstrip("/")

    candidates = [
        # 1. Absolute path stored directly (rare but possible)
        Path(filepath),
        # 2. Relative to the process CWD (may work if worker and server share CWD)
        Path(os.getcwd()) / rel,
        # 3. Relative to upload_dir's resolved parent  (most reliable)
        settings.upload_dir.resolve().parent / rel,
        # 4. upload_dir itself as anchor
        settings.upload_dir.resolve() / rel,
    ]

    for p in candidates:
        try:
            if p.resolve().exists():
                return p.resolve()
        except Exception:
            continue

    return None


def _nms_filter(ann_rows, iou_threshold=0.5):
    """
    Remove overlapping detections via Non-Maximum Suppression on normalised
    xywh boxes.  Keeps the highest-confidence box when two boxes overlap
    beyond *iou_threshold*.  Each entry in *ann_rows* must carry a ``conf``
    key (float).
    """
    if len(ann_rows) <= 1:
        return ann_rows

    # Convert normalised xywh → xyxy for IoU calculation
    boxes = np.array([[
        a["bbox"][0] - a["bbox"][2] / 2,
        a["bbox"][1] - a["bbox"][3] / 2,
        a["bbox"][0] + a["bbox"][2] / 2,
        a["bbox"][1] + a["bbox"][3] / 2,
    ] for a in ann_rows])
    scores = np.array([a["conf"] for a in ann_rows])

    # Sort by confidence descending
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        # IoU of current best vs remaining
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                  (boxes[order[1:], 3] - boxes[order[1:], 1]))
        iou = inter / (area_i + area_j - inter + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return [ann_rows[k] for k in keep]


@celery_app.task(name="app.tasks.auto_annotate.auto_annotate_remaining", bind=True)
def auto_annotate_remaining(self, project_id: str, image_ids: list = None,
                            conf: float = 0.25, use_tta: bool = False):
    """
    Synchronous Celery task — uses StateDBConnector (psycopg2) instead of the
    async SQLAlchemy engine so there is no asyncio event-loop conflict.

    Parameters
    ----------
    conf : float
        Minimum detection confidence (default raised to 0.25 to reduce
        hallucinated annotations).
    use_tta : bool
        Enable Test-Time Augmentation — averages predictions over flipped /
        scaled copies of each image for more robust detections.
    """
    db = StateDBConnector()

    # ── 1. Check seed model (no DB needed) ──────────────────────────
    model_path = settings.model_dir.resolve() / project_id / "seed_best.pt"
    if not model_path.exists():
        return {"error": "Seed model not found. Train the seed model first."}

    model = YOLO(str(model_path))
    class_map = model.names  # {cls_idx: 'class_name', ...}

    # ── 2. Fetch images to annotate ─────────────────────────────────
    with db.get_session() as conn:
        proj_rows = db.execute_query(
            conn,
            "SELECT id FROM projects WHERE id = :project_id",
            {"project_id": project_id},
        )
        if not proj_rows:
            return {"error": "Project not found"}

        if image_ids:
            placeholders = ", ".join(f":id_{i}" for i in range(len(image_ids)))
            id_params = {f"id_{i}": v for i, v in enumerate(image_ids)}
            id_params["project_id"] = project_id
            img_rows = db.execute_query(
                conn,
                f"SELECT id, filename, filepath FROM images "
                f"WHERE id IN ({placeholders}) AND project_id = :project_id",
                id_params,
            )
        else:
            img_rows = db.execute_query(
                conn,
                "SELECT id, filename, filepath FROM images "
                "WHERE project_id = :project_id AND status = 'pending'",
                {"project_id": project_id},
            )
    # connection returned to pool here

    if not img_rows:
        return {"error": "No images to annotate"}

    total = len(img_rows)
    total_annotated = 0
    total_skipped_path = 0   # file not found on disk
    total_no_detection = 0   # file found but model detected nothing

    self.update_state(
        state="STARTED",
        meta={
            "current": 0,
            "total": total,
            "current_image": None,
            "annotated_count": 0,
            "skipped_path": 0,
            "no_detection": 0,
            "conf": conf,
        },
    )

    # ── 3. Predict → save annotations (one connection for all writes) ─
    with db.get_session() as conn:
        for idx, img in enumerate(img_rows):

            label = img.get("filename") or img["filepath"]

            # Resolve filesystem path robustly
            real_path = _resolve_image_path(img["filepath"])

            if real_path is None:
                total_skipped_path += 1
                self.update_state(
                    state="STARTED",
                    meta={
                        "current": idx + 1,
                        "total": total,
                        "current_image": f"[PATH NOT FOUND] {label}",
                        "annotated_count": total_annotated,
                        "skipped_path": total_skipped_path,
                        "no_detection": total_no_detection,
                        "conf": conf,
                    },
                )
                continue

            # Per-image progress
            self.update_state(
                state="STARTED",
                meta={
                    "current": idx + 1,
                    "total": total,
                    "current_image": label,
                    "annotated_count": total_annotated,
                    "skipped_path": total_skipped_path,
                    "no_detection": total_no_detection,
                    "conf": conf,
                },
            )

            # Run YOLO prediction (with optional TTA for robustness)
            try:
                results = model.predict(
                    str(real_path), conf=conf, verbose=False,
                    augment=use_tta,
                )
            except Exception:
                total_skipped_path += 1
                continue

            # Build annotation rows — now storing confidence for filtering
            ann_rows = []
            for r in results:
                for box in r.boxes:
                    cls_idx = int(box.cls[0].cpu().numpy())
                    class_name = class_map.get(cls_idx)
                    if not class_name:
                        continue
                    box_conf = float(box.conf[0].cpu().numpy())
                    xywhn = box.xywhn[0].cpu().numpy().tolist()
                    ann_rows.append({
                        "ann_id":     str(uuid.uuid4()),
                        "image_id":   img["id"],
                        "class_name": class_name,
                        "bbox":       xywhn,
                        "conf":       box_conf,
                        # Pass as JSON string; PostgreSQL casts to JSONB
                        "bbox_json":  json.dumps(xywhn),
                    })

            # Per-class NMS to remove overlapping duplicate detections
            if len(ann_rows) > 1:
                by_class = {}
                for a in ann_rows:
                    by_class.setdefault(a["class_name"], []).append(a)
                ann_rows = []
                for cls_anns in by_class.values():
                    ann_rows.extend(_nms_filter(cls_anns, iou_threshold=0.5))

            if ann_rows:
                db.execute_many(
                    conn,
                    "INSERT INTO annotations (id, image_id, class_name, bbox, source) "
                    "VALUES (:ann_id, :image_id, :class_name, CAST(:bbox_json AS JSONB), 'auto')",
                    ann_rows,
                )
                db.execute_update(
                    conn,
                    "UPDATE images SET status = 'annotated' WHERE id = :id",
                    {"id": img["id"]},
                )
                total_annotated += 1
            else:
                total_no_detection += 1
    # all changes committed here

    return {
        "status": "success",
        "annotated_count": total_annotated,
        "total": total,
        "skipped_path": total_skipped_path,
        "no_detection": total_no_detection,
        "conf_used": conf,
    }
