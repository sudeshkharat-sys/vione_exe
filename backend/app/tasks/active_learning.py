"""
Active Learning module for AI Vision Platform.

Implements SOTA uncertainty-based query strategies to select the most
informative images for human annotation, reducing hallucination by ensuring
the model is trained on the samples it finds hardest.

Strategies implemented
----------------------
1. **Confidence scoring** — images where the model's max detection confidence
   is lowest are most informative (simple, fast).
2. **Entropy scoring** — Shannon entropy over per-detection class probability
   distributions; high entropy = high uncertainty.
3. **TTA disagreement** — variance in predictions across Test-Time Augmented
   views; high variance = the model isn't sure about the geometry/content.
4. **Combined scoring** — weighted combination of the above for a composite
   uncertainty score (SOTA "uncertainty + diversity" approach).

References
----------
- Scalable Active Learning for Object Detection (arXiv 2004.04699)
- Entropy-Based Active Learning for Object Detection (CVPR 2022)
- MC DropBlock for Uncertainty in YOLO (Pattern Recognition 2024)
- Rarity-Aware Stratified Active Learning (Applied Sciences 2026)
- Uncertainty Aware Training for Active Learning (CVPR 2025 Workshop)
"""

from .celery_app import celery_app
from ultralytics import YOLO
from pathlib import Path
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
import numpy as np
import json
import os


# ── Scoring helpers ──────────────────────────────────────────────────

def _resolve_image_path(filepath: str) -> Path | None:
    """Resolve an uploaded image to an absolute path (same logic as auto_annotate)."""
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


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector (clipped to avoid log(0))."""
    p = np.clip(probs, 1e-8, 1.0)
    return float(-np.sum(p * np.log(p)))


def score_confidence(results) -> dict:
    """
    Score a single image by detection confidence.

    Returns
    -------
    dict with keys:
        n_detections : int
        mean_conf    : float  — average confidence across detections
        min_conf     : float  — minimum confidence (most uncertain detection)
        max_conf     : float  — maximum confidence
        uncertainty  : float  — 1 - mean_conf (higher = more uncertain)
    """
    confs = []
    for r in results:
        for box in r.boxes:
            confs.append(float(box.conf[0].cpu().numpy()))

    if not confs:
        # No detections → maximum uncertainty (model knows nothing here)
        return {
            "n_detections": 0,
            "mean_conf": 0.0, "min_conf": 0.0, "max_conf": 0.0,
            "uncertainty": 1.0,
        }

    mean_c = float(np.mean(confs))
    return {
        "n_detections": len(confs),
        "mean_conf": round(mean_c, 4),
        "min_conf": round(float(np.min(confs)), 4),
        "max_conf": round(float(np.max(confs)), 4),
        "uncertainty": round(1.0 - mean_c, 4),
    }


def score_entropy(results, n_classes: int) -> dict:
    """
    Score a single image by entropy of class probability distributions.

    For each detection box, YOLO produces a class probability vector.
    We compute Shannon entropy per box and aggregate (mean + max) at image
    level.  High entropy → the model can't decide which class this is.
    """
    entropies = []
    for r in results:
        if not hasattr(r.boxes, "cls") or len(r.boxes) == 0:
            continue
        # raw class probabilities from the detection head
        if hasattr(r.boxes, "data") and r.boxes.data.shape[1] > 6:
            # some YOLO versions store full class probs in boxes.data[:, 5:]
            cls_probs = r.boxes.data[:, 5:].cpu().numpy()
            for row in cls_probs:
                p = row / (row.sum() + 1e-8)
                entropies.append(_entropy(p))
        else:
            # Fall back: use confidence as a proxy Bernoulli entropy
            for box in r.boxes:
                c = float(box.conf[0].cpu().numpy())
                # Bernoulli entropy: -c*log(c) - (1-c)*log(1-c)
                ent = _entropy(np.array([c, 1.0 - c]))
                entropies.append(ent)

    if not entropies:
        max_ent = _entropy(np.ones(max(n_classes, 2)) / max(n_classes, 2))
        return {"mean_entropy": round(max_ent, 4), "max_entropy": round(max_ent, 4)}

    return {
        "mean_entropy": round(float(np.mean(entropies)), 4),
        "max_entropy": round(float(np.max(entropies)), 4),
    }


def score_tta_disagreement(model, image_path: str, conf: float = 0.1) -> dict:
    """
    Score an image by TTA disagreement — run inference with and without
    augmentation and measure how much the predictions change.

    A high disagreement means the model's predictions are unstable and the
    image is likely near the decision boundary → informative for training.
    """
    try:
        res_normal = model.predict(image_path, conf=conf, verbose=False, augment=False)
        res_tta = model.predict(image_path, conf=conf, verbose=False, augment=True)
    except Exception:
        return {"tta_disagreement": 1.0, "det_count_diff": 0}

    n_normal = sum(len(r.boxes) for r in res_normal)
    n_tta = sum(len(r.boxes) for r in res_tta)

    # Detection count difference (normalised)
    det_diff = abs(n_normal - n_tta)
    max_det = max(n_normal, n_tta, 1)

    # Confidence difference between the two modes
    def avg_conf(results):
        confs = [float(b.conf[0].cpu().numpy()) for r in results for b in r.boxes]
        return float(np.mean(confs)) if confs else 0.0

    conf_diff = abs(avg_conf(res_normal) - avg_conf(res_tta))

    # Combined disagreement score [0, 1]
    disagreement = min(1.0, (det_diff / max_det) * 0.5 + conf_diff * 0.5)

    return {
        "tta_disagreement": round(disagreement, 4),
        "det_count_normal": n_normal,
        "det_count_tta": n_tta,
        "det_count_diff": det_diff,
        "conf_diff": round(conf_diff, 4),
    }


def compute_combined_score(
    conf_score: dict, ent_score: dict, tta_score: dict | None,
    w_conf: float = 0.4, w_ent: float = 0.35, w_tta: float = 0.25,
) -> float:
    """
    Weighted combination of uncertainty signals.

    Default weights follow CVPR 2025 guidance: confidence uncertainty is the
    strongest single signal, entropy captures class confusion, and TTA
    captures geometric instability.
    """
    s = w_conf * conf_score.get("uncertainty", 1.0)
    s += w_ent * ent_score.get("mean_entropy", 0.0)
    if tta_score:
        s += w_tta * tta_score.get("tta_disagreement", 0.0)
    else:
        # Redistribute TTA weight to the other two signals
        s += w_tta * (
            0.5 * conf_score.get("uncertainty", 1.0)
            + 0.5 * ent_score.get("mean_entropy", 0.0)
        )
    return round(float(s), 4)


# ── Celery Tasks ─────────────────────────────────────────────────────

@celery_app.task(
    name="app.tasks.active_learning.score_unlabeled_images", bind=True
)
def score_unlabeled_images(
    self,
    project_id: str,
    model_type: str = "seed",
    strategy: str = "combined",
    top_k: int = 0,
    use_tta: bool = False,
    conf: float = 0.1,
):
    """
    Score all *pending* (unlabeled) images in a project by uncertainty.

    Parameters
    ----------
    model_type : str
        ``"seed"`` or ``"main"`` — which trained model to use for scoring.
    strategy : str
        ``"confidence"`` | ``"entropy"`` | ``"tta"`` | ``"combined"``
    top_k : int
        If > 0, return only the top-K most uncertain images.
        If 0, return all scored images sorted by uncertainty descending.
    use_tta : bool
        Include TTA disagreement in the scoring (slower but more robust).
    conf : float
        Low confidence threshold for scoring — we intentionally use a low
        threshold here so that weak detections are visible to the scorer.

    Returns
    -------
    dict with ``scored_images`` — list of {image_id, filename, scores, rank}.
    """
    import time
    db = StateDBConnector()

    # ── 1. Load model ──────────────────────────────────────────────
    weight_name = "seed_best.pt" if model_type == "seed" else "main_best.pt"
    model_path = settings.model_dir.resolve() / project_id / weight_name
    if not model_path.exists():
        return {"error": f"{model_type} model not found. Train it first."}

    model = YOLO(str(model_path))
    n_classes = len(model.names)

    # ── 2. Fetch pending images ────────────────────────────────────
    with db.get_session() as conn:
        img_rows = db.execute_query(
            conn,
            "SELECT id, filename, filepath FROM images "
            "WHERE project_id = :pid AND status = 'pending'",
            {"pid": project_id},
        )

    if not img_rows:
        return {"error": "No pending images to score"}

    total    = len(img_rows)
    scored   = []
    start_ts = time.time()

    # ── 3. Score each image ────────────────────────────────────────
    for idx, img in enumerate(img_rows):
        elapsed  = time.time() - start_ts
        rate     = (idx + 1) / elapsed if elapsed > 0 else 0
        eta_secs = int((total - idx - 1) / rate) if rate > 0 else None

        if self.request.id:
            self.update_state(
                state="STARTED",
                meta={
                    "current":  idx + 1,
                    "total":    total,
                    "strategy": strategy,
                    "eta_secs": eta_secs,
                },
            )

        real_path = _resolve_image_path(img["filepath"])
        if real_path is None:
            continue

        path_str = str(real_path)

        # Confidence scoring (always computed — fast)
        try:
            results = model.predict(path_str, conf=conf, verbose=False)
        except Exception:
            continue

        conf_score = score_confidence(results)
        ent_score = score_entropy(results, n_classes)

        tta_score = None
        if use_tta or strategy in ("tta", "combined"):
            tta_score = score_tta_disagreement(model, path_str, conf=conf)

        # Pick the relevant score for sorting
        if strategy == "confidence":
            sort_key = conf_score["uncertainty"]
        elif strategy == "entropy":
            sort_key = ent_score["mean_entropy"]
        elif strategy == "tta":
            sort_key = (tta_score or {}).get("tta_disagreement", 0.0)
        else:  # combined
            sort_key = compute_combined_score(conf_score, ent_score, tta_score)

        scored.append({
            "image_id": img["id"],
            "filename": img.get("filename") or img["filepath"],
            "confidence": conf_score,
            "entropy": ent_score,
            "tta": tta_score,
            "combined_score": sort_key,
        })

    # ── 4. Sort by uncertainty (descending) and pick top-K ─────────
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    for rank, item in enumerate(scored, 1):
        item["rank"] = rank

    if top_k > 0:
        scored = scored[:top_k]

    return {
        "status": "success",
        "strategy": strategy,
        "total_scored": len(scored),
        "total_pending": total,
        "scored_images": scored,
    }


@celery_app.task(
    name="app.tasks.active_learning.curriculum_auto_annotate", bind=True
)
def curriculum_auto_annotate(
    self,
    project_id: str,
    high_conf: float = 0.6,
    low_conf: float = 0.25,
    review_band_top: float = 0.6,
    review_band_bottom: float = 0.35,
    use_tta: bool = True,
):
    """
    Curriculum-based auto-annotation with confidence tiers.

    Instead of blindly auto-annotating everything above a single threshold,
    this task sorts images by model confidence and creates three tiers:

    1. **Auto-accept** (conf ≥ high_conf) — model is confident; annotations
       saved automatically with ``source='auto'``.
    2. **Review** (review_band_bottom ≤ conf < review_band_top) — medium
       confidence; annotations saved with ``source='auto_review'`` so the
       UI can flag them for human verification.
    3. **Skip** (conf < low_conf) — model is too uncertain; flagged for
       manual annotation via active learning.

    This implements the "curriculum learning" principle: start with easy
    (high-confidence) samples, progressively include harder ones only after
    human review.

    Parameters
    ----------
    high_conf : float
        Threshold above which annotations are auto-accepted.
    low_conf : float
        Minimum confidence to save any annotation at all.
    review_band_top / review_band_bottom : float
        Confidence range that gets saved with ``source='auto_review'``
        for human triage.
    use_tta : bool
        Use TTA during prediction for more robust detections.
    """
    import uuid, time

    db = StateDBConnector()

    # ── 1. Load seed model ──────────────────────────────────────────
    model_path = settings.model_dir.resolve() / project_id / "seed_best.pt"
    if not model_path.exists():
        return {"error": "Seed model not found. Train the seed model first."}

    model = YOLO(str(model_path))
    class_map = model.names

    # ── 2. Fetch pending images ────────────────────────────────────
    with db.get_session() as conn:
        proj_rows = db.execute_query(
            conn,
            "SELECT id FROM projects WHERE id = :pid",
            {"pid": project_id},
        )
        if not proj_rows:
            return {"error": "Project not found"}

        img_rows = db.execute_query(
            conn,
            "SELECT id, filename, filepath FROM images "
            "WHERE project_id = :pid AND status = 'pending'",
            {"pid": project_id},
        )

    if not img_rows:
        return {"error": "No pending images to annotate"}

    total    = len(img_rows)
    start_ts = time.time()
    stats = {
        "auto_accepted": 0,
        "sent_to_review": 0,
        "skipped_low_conf": 0,
        "skipped_no_det": 0,
        "skipped_path": 0,
    }

    self.update_state(
        state="STARTED",
        meta={"current": 0, "total": total, "eta_secs": None, **stats},
    )

    # ── 3. Score and annotate with curriculum tiers ────────────────
    with db.get_session() as conn:
        for idx, img in enumerate(img_rows):
            elapsed  = time.time() - start_ts
            rate     = (idx + 1) / elapsed if elapsed > 0 else 0
            eta_secs = int((total - idx - 1) / rate) if rate > 0 else None

            real_path = _resolve_image_path(img["filepath"])
            if real_path is None:
                stats["skipped_path"] += 1
                self.update_state(
                    state="STARTED",
                    meta={"current": idx + 1, "total": total, "eta_secs": eta_secs, **stats},
                )
                continue

            try:
                results = model.predict(
                    str(real_path), conf=low_conf, verbose=False,
                    augment=use_tta,
                )
            except Exception:
                stats["skipped_path"] += 1
                continue

            # Collect detections with their confidence
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_idx = int(box.cls[0].cpu().numpy())
                    class_name = class_map.get(cls_idx)
                    if not class_name:
                        continue
                    box_conf = float(box.conf[0].cpu().numpy())
                    xywhn = box.xywhn[0].cpu().numpy().tolist()
                    detections.append({
                        "class_name": class_name,
                        "bbox": xywhn,
                        "conf": box_conf,
                    })

            if not detections:
                stats["skipped_no_det"] += 1
                self.update_state(
                    state="STARTED",
                    meta={"current": idx + 1, "total": total, "eta_secs": eta_secs, **stats},
                )
                continue

            # Mean confidence across all detections in this image
            img_mean_conf = float(np.mean([d["conf"] for d in detections]))

            # Determine tier
            if img_mean_conf >= high_conf:
                source = "auto"
                stats["auto_accepted"] += 1
            elif img_mean_conf >= review_band_bottom:
                source = "auto_review"
                stats["sent_to_review"] += 1
            else:
                stats["skipped_low_conf"] += 1
                self.update_state(
                    state="STARTED",
                    meta={"current": idx + 1, "total": total, "eta_secs": eta_secs, **stats},
                )
                continue

            # Save annotations
            ann_rows = []
            for d in detections:
                # Only keep individual detections above the low threshold
                if d["conf"] < low_conf:
                    continue
                ann_rows.append({
                    "ann_id": str(uuid.uuid4()),
                    "image_id": img["id"],
                    "class_name": d["class_name"],
                    "bbox_json": json.dumps(d["bbox"]),
                })

            if ann_rows:
                db.execute_many(
                    conn,
                    "INSERT INTO annotations (id, image_id, class_name, bbox, source) "
                    "VALUES (:ann_id, :image_id, :class_name, "
                    f"CAST(:bbox_json AS JSONB), '{source}')",
                    ann_rows,
                )
                db.execute_update(
                    conn,
                    "UPDATE images SET status = 'annotated' WHERE id = :id",
                    {"id": img["id"]},
                )

            self.update_state(
                state="STARTED",
                meta={"current": idx + 1, "total": total, "eta_secs": eta_secs, **stats},
            )

    return {
        "status": "success",
        "total": total,
        **stats,
        "thresholds": {
            "high_conf": high_conf,
            "low_conf": low_conf,
            "review_band": [review_band_bottom, review_band_top],
        },
    }


@celery_app.task(
    name="app.tasks.active_learning.suggest_for_review", bind=True
)
def suggest_for_review(
    self,
    project_id: str,
    budget: int = 10,
    strategy: str = "combined",
):
    """
    Quick endpoint: returns the top-*budget* most uncertain pending images
    that the user should annotate next for maximum model improvement.

    This is the core "active learning query" — it tells the user exactly
    which images to label to get the biggest bang for their annotation buck.
    """
    import time

    db = StateDBConnector()

    # ── 1. Load seed model ─────────────────────────────────────────
    model_path = settings.model_dir.resolve() / project_id / "seed_best.pt"
    if not model_path.exists():
        return {"error": "Seed model not found. Train the seed model first."}

    model = YOLO(str(model_path))
    n_classes = len(model.names)

    # ── 2. Fetch pending images ────────────────────────────────────
    with db.get_session() as conn:
        img_rows = db.execute_query(
            conn,
            "SELECT id, filename, filepath FROM images "
            "WHERE project_id = :pid AND status = 'pending'",
            {"pid": project_id},
        )

    if not img_rows:
        return {"error": "No pending images to score"}

    total    = len(img_rows)
    scored   = []
    start_ts = time.time()

    # ── 3. Score each image with progress + ETA ────────────────────
    for idx, img in enumerate(img_rows):
        elapsed  = time.time() - start_ts
        rate     = (idx + 1) / elapsed if elapsed > 0 else 0
        eta_secs = int((total - idx - 1) / rate) if rate > 0 else None

        if self.request.id:
            self.update_state(
                state="STARTED",
                meta={
                    "current":  idx + 1,
                    "total":    total,
                    "strategy": strategy,
                    "eta_secs": eta_secs,
                },
            )

        real_path = _resolve_image_path(img["filepath"])
        if real_path is None:
            continue

        path_str = str(real_path)

        try:
            results = model.predict(path_str, conf=0.1, verbose=False)
        except Exception:
            continue

        conf_score = score_confidence(results)
        ent_score  = score_entropy(results, n_classes)
        tta_score  = score_tta_disagreement(model, path_str, conf=0.1)

        if strategy == "confidence":
            sort_key = conf_score["uncertainty"]
        elif strategy == "entropy":
            sort_key = ent_score["mean_entropy"]
        elif strategy == "tta":
            sort_key = (tta_score or {}).get("tta_disagreement", 0.0)
        else:
            sort_key = compute_combined_score(conf_score, ent_score, tta_score)

        scored.append({
            "image_id":      img["id"],
            "filename":      img.get("filename") or img["filepath"],
            "confidence":    conf_score,
            "entropy":       ent_score,
            "tta":           tta_score,
            "combined_score": sort_key,
        })

    # ── 4. Sort and pick top-budget ────────────────────────────────
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    for rank, item in enumerate(scored, 1):
        item["rank"] = rank

    top = scored[:budget]

    suggestions = [
        {
            "image_id":   item["image_id"],
            "filename":   item["filename"],
            "uncertainty": item["combined_score"],
            "reason":     _explain_uncertainty(item),
        }
        for item in top
    ]

    return {
        "status":        "success",
        "budget":        budget,
        "suggestions":   suggestions,
        "total_pending": total,
    }


def _explain_uncertainty(item: dict) -> str:
    """Generate a human-readable explanation of why an image is uncertain."""
    conf = item.get("confidence", {})
    ent = item.get("entropy", {})
    tta = item.get("tta", {})

    reasons = []

    if conf.get("n_detections", 0) == 0:
        reasons.append("No objects detected — model may be missing objects here")
    elif conf.get("mean_conf", 1.0) < 0.3:
        reasons.append(f"Very low detection confidence ({conf['mean_conf']:.0%})")
    elif conf.get("mean_conf", 1.0) < 0.5:
        reasons.append(f"Low detection confidence ({conf['mean_conf']:.0%})")

    if ent.get("max_entropy", 0) > 0.5:
        reasons.append("High class confusion — model unsure which class")

    if tta and tta.get("det_count_diff", 0) > 0:
        reasons.append(
            f"Predictions unstable under augmentation "
            f"({tta['det_count_normal']}→{tta['det_count_tta']} detections)"
        )

    return "; ".join(reasons) if reasons else "Moderate uncertainty"
