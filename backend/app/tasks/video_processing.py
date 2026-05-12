"""
video_processing.py
~~~~~~~~~~~~~~~~~~~
Celery task that extracts frames from an uploaded video file and registers
each frame as an Image row so it flows into the normal annotation pipeline.

How it works
------------
1.  The FastAPI endpoint triggers ``extract_video_frames.delay(video_id, ...)``.
2.  The worker opens the video with OpenCV, seeks to each sampled frame,
    saves it as a JPEG, and inserts an Image row into PostgreSQL via the
    synchronous StateDBConnector (same pattern as training.py / auto_annotate.py).
3.  After all frames are saved the Video row status is updated to ``"done"``.
    On any unhandled exception it is set to ``"failed"``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
from loguru import logger

from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
from .celery_app import celery_app


def _resolve_video_path(filepath: str) -> Path | None:
    """Try several anchors to find the video file on disk."""
    rel = filepath.lstrip("/")
    candidates = [
        Path(filepath),
        Path(os.getcwd()) / rel,
        settings.upload_dir.resolve().parent / rel,
        settings.upload_dir.resolve() / rel,
    ]
    for p in candidates:
        try:
            if p.resolve().exists():
                return p.resolve()
        except Exception:
            continue
    return None


@celery_app.task(bind=True, name="extract_video_frames")
def extract_video_frames(
    self,
    video_id: str,
    sample_every_n: int = 30,
    max_frames: int = 300,
) -> dict:
    """
    Extract frames from a video and create Image records for each one.

    Parameters
    ----------
    video_id:
        UUID of the Video row to process.
    sample_every_n:
        Keep 1 out of every N frames  (e.g. 30 → ~1 fps for a 30-fps video).
    max_frames:
        Hard cap on number of frames extracted.  0 means no limit.
    """
    db = StateDBConnector()

    try:
        # ── 1. Load the Video row ─────────────────────────────────────────────
        with db.get_session() as conn:
            video_rows = db.execute_query(
                conn,
                "SELECT id, project_id, filepath, original_filename FROM videos WHERE id = :vid",
                {"vid": video_id},
            )

        if not video_rows:
            logger.error(f"[video] Video {video_id} not found in DB.")
            return {"status": "failed", "reason": "video not found"}

        video_row = video_rows[0]
        project_id = video_row["project_id"]
        filepath_str = video_row["filepath"]
        original_filename = video_row["original_filename"]

        # ── 2. Mark as extracting ─────────────────────────────────────────────
        with db.get_session() as conn:
            db.execute_update(
                conn,
                "UPDATE videos SET status = 'extracting' WHERE id = :vid",
                {"vid": video_id},
            )

        # ── 3. Locate the file ────────────────────────────────────────────────
        video_path = _resolve_video_path(filepath_str)
        if video_path is None:
            logger.error(f"[video] File not found on disk: {filepath_str}")
            with db.get_session() as conn:
                db.execute_update(
                    conn,
                    "UPDATE videos SET status = 'failed' WHERE id = :vid",
                    {"vid": video_id},
                )
            return {"status": "failed", "reason": "file not found on disk"}

        # ── 4. Open with OpenCV ───────────────────────────────────────────────
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"[video] cv2 could not open: {video_path}")
            with db.get_session() as conn:
                db.execute_update(
                    conn,
                    "UPDATE videos SET status = 'failed' WHERE id = :vid",
                    {"vid": video_id},
                )
            return {"status": "failed", "reason": "cv2 could not open file"}

        total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames_cv / fps if fps > 0 else 0.0

        # Update video metadata now that we have it from cv2
        with db.get_session() as conn:
            db.execute_update(
                conn,
                """UPDATE videos
                   SET fps = :fps, width = :w, height = :h,
                       total_frames = :tf, duration = :dur
                   WHERE id = :vid""",
                {"fps": fps, "w": vid_w, "h": vid_h,
                 "tf": total_frames_cv, "dur": duration, "vid": video_id},
            )

        # ── 5. Prepare output directory ───────────────────────────────────────
        frames_dir = settings.upload_dir / project_id / "video_frames" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(original_filename).stem  # e.g. "clip_01"

        # ── 6. Extract frames ─────────────────────────────────────────────────
        frame_idx = 0
        saved_count = 0
        effective_max = max_frames if max_frames > 0 else float("inf")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                if saved_count >= effective_max:
                    break

                frame_uuid = str(uuid.uuid4())
                frame_filename = f"{frame_uuid}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # Relative URL path served by the /uploads static mount
                rel_path = (
                    f"/uploads/{project_id}/video_frames/{video_id}/{frame_filename}"
                )
                # Human-readable display name: stem_frame_000042.jpg
                display_name = f"{stem}_frame_{frame_idx:06d}.jpg"

                with db.get_session() as conn:
                    db.execute_insert(
                        conn,
                        """INSERT INTO images
                               (id, project_id, filename, filepath, width, height,
                                status, created_at)
                           VALUES
                               (:id, :pid, :fname, :fpath, :w, :h, 'pending', NOW())""",
                        {
                            "id": frame_uuid,
                            "pid": project_id,
                            "fname": display_name,
                            "fpath": rel_path,
                            "w": vid_w,
                            "h": vid_h,
                        },
                    )

                saved_count += 1

                # Persist progress every 25 frames so the UI can show live count
                if saved_count % 25 == 0:
                    with db.get_session() as conn:
                        db.execute_update(
                            conn,
                            "UPDATE videos SET frames_extracted = :cnt WHERE id = :vid",
                            {"cnt": saved_count, "vid": video_id},
                        )
                    logger.info(
                        f"[video] {video_id}: extracted {saved_count} frames so far…"
                    )

            frame_idx += 1

        cap.release()

        # ── 7. Final status update ────────────────────────────────────────────
        with db.get_session() as conn:
            db.execute_update(
                conn,
                """UPDATE videos
                   SET status = 'done', frames_extracted = :cnt
                   WHERE id = :vid""",
                {"cnt": saved_count, "vid": video_id},
            )

        logger.info(
            f"[video] Done. {saved_count} frames extracted from {original_filename}."
        )
        return {"status": "done", "frames_extracted": saved_count}

    except Exception as exc:
        logger.exception(f"[video] extract_video_frames failed for {video_id}: {exc}")
        try:
            with db.get_session() as conn:
                db.execute_update(
                    conn,
                    "UPDATE videos SET status = 'failed' WHERE id = :vid",
                    {"vid": video_id},
                )
        except Exception:
            pass
        raise
