from .celery_app import celery_app
from ultralytics import YOLO
import os
import shutil
import time
import random
import math
from pathlib import Path
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector
from collections import defaultdict
import yaml
import json
import cv2
import numpy as np
import redis as redis_lib


def _resolve_model_path(model_name: str) -> str:
    """
    Return the full path to a pre-downloaded YOLO base weight if it exists in
    yolo_weights_dir, otherwise fall back to model_name so ultralytics can
    attempt an online download (useful in dev environments with internet).
    """
    preloaded = settings.yolo_weights_dir / model_name
    if preloaded.exists() and preloaded.stat().st_size > 1024 * 1024:
        return str(preloaded)
    return model_name


def _safe_float(v):
    """Convert a numeric value to a JSON-safe float (None for NaN/Inf)."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except Exception:
        return None


def _preprocess_for_inspection(src_path: Path, dst_path: Path) -> None:
    """
    Three-stage preprocessing pipeline tuned for water-pipe clip inspection.

    Stage 1 — Aggressive CLAHE
      clipLimit=4.0, tileGridSize=(4,4): smaller tiles mean tighter local
      adaptation, so the white clip region is enhanced independently of the
      surrounding dark rubber.  Higher clip limit allows more contrast gain
      before clamping, making bright clip edges genuinely white rather than
      just 'less dark'.

    Stage 2 — Gamma correction (γ = 1.3)
      γ > 1 darkens the shadow/midtone range, keeping the dark rubber
      background dark.  This widens the perceived gap between the black
      hose and the white plastic clip — the opposite of γ < 1 which
      lifted dark areas and turned the rubber a flat grey.

    Stage 3 — Unsharp mask sharpening
      Subtracts a Gaussian-blurred copy from the original (weighted sum).
      This crisp-ens the clip-to-rubber boundary — the hard edge between
      white plastic and black hose is exactly the signal the model needs
      to detect.

    Falls back to a plain file copy if OpenCV cannot read the image.
    """
    img = cv2.imread(str(src_path))
    if img is None:
        shutil.copy(src_path, dst_path)
        return

    # ── Stage 1: moderate CLAHE on L channel ─────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # ── Stage 2: gamma correction (γ=1.3) ───────────────────────
    lut = np.array([(i / 255.0) ** 1.3 * 255 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(out, lut)

    # ── Stage 3: unsharp mask sharpening ─────────────────────────
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2.0)
    out = cv2.addWeighted(out, 1.4, blurred, -0.4, 0)

    cv2.imwrite(str(dst_path), out)


# ── Shared helpers ────────────────────────────────────────────────

def _fetch_training_data(db, conn, project_id: str, status_filter: str = "annotated"):
    """Read project + images (filtered by status) + all annotations in one session."""
    proj_rows = db.execute_query(
        conn,
        "SELECT id, classes FROM projects WHERE id = :project_id",
        {"project_id": project_id},
    )
    if not proj_rows:
        return None, None, None, None

    raw_classes = proj_rows[0].get("classes")
    if isinstance(raw_classes, str):
        classes = json.loads(raw_classes) if raw_classes else []
    elif isinstance(raw_classes, list):
        classes = raw_classes
    else:
        classes = []

    img_rows = db.execute_query(
        conn,
        "SELECT id, filename, filepath FROM images "
        "WHERE project_id = :project_id AND status = :status",
        {"project_id": project_id, "status": status_filter},
    )
    if not img_rows:
        return proj_rows[0], classes, [], []

    image_ids = [img["id"] for img in img_rows]

    if not classes:
        placeholders = ", ".join(f":id_{i}" for i in range(len(image_ids)))
        params = {f"id_{i}": v for i, v in enumerate(image_ids)}
        class_rows = db.execute_query(
            conn,
            f"SELECT DISTINCT class_name FROM annotations "
            f"WHERE image_id IN ({placeholders})",
            params,
        )
        classes = [row["class_name"] for row in class_rows]

    placeholders = ", ".join(f":id_{i}" for i in range(len(image_ids)))
    params = {f"id_{i}": v for i, v in enumerate(image_ids)}
    ann_rows = db.execute_query(
        conn,
        f"SELECT image_id, class_name, bbox, source FROM annotations "
        f"WHERE image_id IN ({placeholders})",
        params,
    )
    return proj_rows[0], classes, img_rows, ann_rows


def _group_annotations(ann_rows):
    """Group raw annotation rows by image_id, normalising bbox type."""
    anns_by_image = defaultdict(list)
    for row in ann_rows:
        raw_bbox = row.get("bbox")
        bbox = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
        anns_by_image[row["image_id"]].append({
            "class_name": row["class_name"],
            "bbox": bbox,
            "source": row.get("source", "manual"),
        })
    return anns_by_image


def _classify_image_quality(anns: list) -> str:
    sources = {a.get("source", "manual") for a in anns}
    if "manual" in sources:
        return "manual"
    if "auto_review" in sources:
        return "auto_review"
    return "auto_high"


def _split_images(img_rows, train_ratio=0.8, val_ratio=0.15, seed=42,
                   anns_by_image=None):
    imgs = list(img_rows)
    rng = random.Random(seed)
    n = len(imgs)

    if n < 5:
        rng.shuffle(imgs)
        return imgs, imgs, []

    if anns_by_image:
        quality_order = {"manual": 0, "auto_high": 1, "auto_review": 2}
        imgs.sort(
            key=lambda im: quality_order.get(
                _classify_image_quality(anns_by_image.get(im["id"], [])), 1
            )
        )
        manual_end = 0
        for i, im in enumerate(imgs):
            q = _classify_image_quality(anns_by_image.get(im["id"], []))
            if q != "manual":
                manual_end = i
                break
        else:
            manual_end = n

        manual_imgs = imgs[:manual_end]
        rest_imgs = imgs[manual_end:]
        rng.shuffle(manual_imgs)
        rng.shuffle(rest_imgs)
        imgs = manual_imgs + rest_imgs
    else:
        rng.shuffle(imgs)

    n_train = max(1, round(n * train_ratio))

    if n < 10:
        n_val = n - n_train
        return imgs[:n_train], imgs[n_train:], []

    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train)

    return imgs[:n_train], imgs[n_train:n_train + n_val], imgs[n_train + n_val:]


def _write_split(dataset_path, split_name, split_imgs, anns_by_image, classes,
                 preprocess: bool = True,
                 task=None, progress_offset: int = 0, progress_total: int = 0):
    """Copy (and optionally CLAHE-enhance) images and write label files for one split."""
    (dataset_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(split_imgs):
        real_path = Path(".") / img["filepath"].lstrip("/")
        if not real_path.exists():
            real_path = settings.upload_dir.parent / Path(img["filepath"].lstrip("/"))

        dest_name = os.path.basename(img["filepath"])
        dest_path = dataset_path / "images" / split_name / dest_name

        if preprocess:
            _preprocess_for_inspection(real_path, dest_path)
        else:
            shutil.copy(real_path, dest_path)

        if task and preprocess and progress_total > 0 and (idx + 1) % 5 == 0:
            current = progress_offset + idx + 1
            try:
                task.update_state(
                    state="STARTED",
                    meta={
                        "phase": "preprocessing",
                        "current": current,
                        "total": progress_total,
                        "split": split_name,
                        "pct": round(current / progress_total * 100),
                    },
                )
            except Exception:
                pass

        label_file = (
            dataset_path / "labels" / split_name
            / (os.path.splitext(dest_name)[0] + ".txt")
        )
        with open(label_file, "w") as f:
            for ann in anns_by_image.get(img["id"], []):
                if ann["bbox"] and ann["class_name"] in classes:
                    cls_idx = classes.index(ann["class_name"])
                    bbox = ann["bbox"]
                    f.write(f"{cls_idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def _build_yolo_dataset(img_rows, anns_by_image, classes, project_id,
                        train_ratio=0.8, val_ratio=0.15, preprocess=True, task=None):
    dataset_path = Path(f"./temp_dataset_{project_id}")
    dataset_path.mkdir(exist_ok=True)

    train_imgs, val_imgs, test_imgs = _split_images(
        img_rows, train_ratio=train_ratio, val_ratio=val_ratio,
        anns_by_image=anns_by_image,
    )

    total = len(train_imgs) + len(val_imgs) + len(test_imgs)

    if task and preprocess and total > 0:
        try:
            task.update_state(
                state="STARTED",
                meta={"phase": "preprocessing", "current": 0, "total": total, "split": "train", "pct": 0},
            )
        except Exception:
            pass

    _write_split(dataset_path, "train", train_imgs, anns_by_image, classes,
                 preprocess=preprocess, task=task,
                 progress_offset=0, progress_total=total)
    _write_split(dataset_path, "val",   val_imgs,   anns_by_image, classes,
                 preprocess=preprocess, task=task,
                 progress_offset=len(train_imgs), progress_total=total)
    if test_imgs:
        _write_split(dataset_path, "test", test_imgs, anns_by_image, classes,
                     preprocess=preprocess, task=task,
                     progress_offset=len(train_imgs) + len(val_imgs), progress_total=total)

    data_yaml: dict = {
        "path":  str(dataset_path.absolute()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(classes),
        "names": classes,
    }
    if test_imgs:
        data_yaml["test"] = "images/test"

    with open(dataset_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    return dataset_path, len(train_imgs), len(val_imgs), len(test_imgs)


def _make_epoch_callback(celery_task, total_epochs, epoch_history, epoch_start_times, stop_flag):
    """Return an on_fit_epoch_end callback that pushes live metrics to Celery."""
    def on_fit_epoch_end(trainer):
        # Check Redis stop flag — set by cancel/force-stop-all endpoints
        try:
            _r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=1)
            if _r.get(f"stop_training:{celery_task.request.id}"):
                _r.delete(f"stop_training:{celery_task.request.id}")
                stop_flag['value'] = True
                trainer.stop = True
                return
        except Exception:
            pass

        epoch = trainer.epoch + 1

        losses = {}
        try:
            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                vals = trainer.loss_items
                vals = vals.tolist() if hasattr(vals, "tolist") else list(vals)
                names = getattr(trainer, "loss_names", ["box_loss", "cls_loss", "dfl_loss"])
                for name, v in zip(names, vals):
                    losses[name] = _safe_float(v)
        except Exception:
            pass

        metrics = {}
        try:
            if hasattr(trainer, "metrics") and trainer.metrics:
                for k, v in trainer.metrics.items():
                    clean = k.replace("metrics/", "").replace("(B)", "")
                    metrics[clean] = _safe_float(v)
        except Exception:
            pass

        now = time.time()
        epoch_start_times.append(now)
        eta_seconds = None
        if len(epoch_start_times) >= 2:
            avg = (epoch_start_times[-1] - epoch_start_times[0]) / max(
                len(epoch_start_times) - 1, 1
            )
            eta_seconds = round(avg * (total_epochs - epoch))

        entry = {"epoch": epoch, **losses, **metrics}
        epoch_history.append(entry)

        try:
            celery_task.update_state(
                state="STARTED",
                meta={
                    "epoch":        epoch,
                    "total_epochs": total_epochs,
                    "eta_seconds":  eta_seconds,
                    "history":      epoch_history,
                },
            )
        except Exception:
            pass

    return on_fit_epoch_end


# ══════════════════════════════════════════════════════════════════
#  Seed Training Task
# ══════════════════════════════════════════════════════════════════

@celery_app.task(name="app.tasks.training.train_seed_model", bind=True)
def train_seed_model(
    self,
    project_id: str,
    model_name: str = "yolo11s.pt",
    epochs: int = 40,
    imgsz: int = 640,
    preprocess: bool = True,
    batch: int = -1,
    custom_weights: str = None,
    aug_fliplr: float = 0.5,
    aug_flipud: float = 0.1,
    aug_mosaic: float = 0.5,
    aug_hsv_v: float = 0.4,
    aug_hsv_h: float = 0.015,
    aug_hsv_s: float = 0.3,
    aug_degrees: float = 10.0,
    aug_translate: float = 0.1,
    aug_scale: float = 0.4,
    aug_mixup: float = 0.0,
    aug_copy_paste: float = 0.05,
):
    db = StateDBConnector()

    with db.get_session() as conn:
        proj, classes, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated"
        )

    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}

    anns_by_image = _group_annotations(ann_rows)

    dataset_path, n_train, n_val, n_test = _build_yolo_dataset(
        img_rows, anns_by_image, classes, project_id,
        preprocess=preprocess, task=self,
    )

    total_epochs   = epochs
    epoch_history  = []
    epoch_start_times = []
    stop_flag      = {'value': False}

    if custom_weights:
        custom_path = settings.model_dir / project_id / "custom_weights" / custom_weights
        if not custom_path.exists():
            return {"error": f"Uploaded weights '{custom_weights}' not found."}
        model = YOLO(str(custom_path))
    else:
        model = YOLO(_resolve_model_path(model_name))

    model.add_callback(
        "on_fit_epoch_end",
        _make_epoch_callback(self, total_epochs, epoch_history, epoch_start_times, stop_flag),
    )

    self.update_state(
        state="STARTED",
        meta={"epoch": 0, "total_epochs": total_epochs, "eta_seconds": None,
              "history": [], "model_name": custom_weights or model_name,
              "split": {"train": n_train, "val": n_val, "test": n_test}},
    )

    _batch = 0.90 if batch == -1 else batch

    results = model.train(
        data=str(dataset_path / "data.yaml"),
        epochs=total_epochs,
        imgsz=imgsz,
        batch=_batch,
        cache=True,
        amp=True,
        device=0,
        lr0=settings.seed_learning_rate,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.001,
        patience=20,
        label_smoothing=0.1,
        hsv_h=aug_hsv_h,
        hsv_s=aug_hsv_s,
        hsv_v=aug_hsv_v,
        degrees=aug_degrees,
        translate=aug_translate,
        scale=aug_scale,
        fliplr=aug_fliplr,
        flipud=aug_flipud,
        mosaic=aug_mosaic,
        close_mosaic=15,
        mixup=aug_mixup,
        copy_paste=aug_copy_paste,
        project=str(settings.model_dir / project_id),
        name="seed_model",
        verbose=False,
        workers=0,
    )

    if stop_flag['value']:
        try:
            shutil.rmtree(results.save_dir, ignore_errors=True)
        except Exception:
            pass
        shutil.rmtree(dataset_path, ignore_errors=True)
        from celery.exceptions import Ignore
        raise Ignore()

    best_model_path = results.save_dir / "weights" / "best.pt"
    target_path = settings.model_dir / project_id / "seed_best.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, target_path)
    shutil.rmtree(dataset_path)

    final_metrics = epoch_history[-1] if epoch_history else {}

    return {
        "status":     "success",
        "model_path": str(target_path),
        "model_name": model_name,
        "metrics":    final_metrics,
        "history":    epoch_history,
        "split":      {"train": n_train, "val": n_val, "test": n_test},
    }


# ══════════════════════════════════════════════════════════════════
#  Main Training Task
# ══════════════════════════════════════════════════════════════════

@celery_app.task(name="app.tasks.training.train_main_model", bind=True)
def train_main_model(
    self,
    project_id: str,
    model_name: str = "yolo11s.pt",
    epochs: int = 60,
    use_seed_weights: bool = True,
    imgsz: int = 640,
    preprocess: bool = True,
    batch: int = -1,
    custom_weights: str = None,
    aug_fliplr: float = 0.5,
    aug_flipud: float = 0.1,
    aug_mosaic: float = 0.5,
    aug_hsv_v: float = 0.4,
    aug_hsv_h: float = 0.015,
    aug_hsv_s: float = 0.3,
    aug_degrees: float = 10.0,
    aug_translate: float = 0.1,
    aug_scale: float = 0.4,
    aug_mixup: float = 0.0,
    aug_copy_paste: float = 0.1,
):
    db = StateDBConnector()

    with db.get_session() as conn:
        proj, classes, img_rows, ann_rows = _fetch_training_data(
            db, conn, project_id, status_filter="annotated"
        )

    if proj is None:
        return {"error": "Project not found"}
    if not img_rows:
        return {"error": "No annotated images found"}

    if custom_weights:
        custom_path = settings.model_dir / project_id / "custom_weights" / custom_weights
        if not custom_path.exists():
            return {"error": f"Uploaded weights '{custom_weights}' not found."}
        pretrained = str(custom_path)
    elif use_seed_weights:
        seed_path = settings.model_dir / project_id / "seed_best.pt"
        if not seed_path.exists():
            return {"error": "Seed model not found — train seed model first, or disable 'Use seed weights'."}
        pretrained = str(seed_path)
    else:
        pretrained = _resolve_model_path(model_name)

    anns_by_image = _group_annotations(ann_rows)

    dataset_path, n_train, n_val, n_test = _build_yolo_dataset(
        img_rows, anns_by_image, classes, f"{project_id}_main",
        preprocess=preprocess, task=self,
    )

    total_epochs   = epochs
    epoch_history  = []
    epoch_start_times = []
    stop_flag      = {'value': False}

    lr0 = (
        settings.main_learning_rate / 2
        if use_seed_weights
        else settings.main_learning_rate
    )

    model = YOLO(pretrained)
    model.add_callback(
        "on_fit_epoch_end",
        _make_epoch_callback(self, total_epochs, epoch_history, epoch_start_times, stop_flag),
    )

    self.update_state(
        state="STARTED",
        meta={"epoch": 0, "total_epochs": total_epochs, "eta_seconds": None,
              "history": [], "model_name": model_name,
              "use_seed_weights": use_seed_weights,
              "split": {"train": n_train, "val": n_val, "test": n_test}},
    )

    _batch = 0.90 if batch == -1 else batch

    results = model.train(
        data=str(dataset_path / "data.yaml"),
        epochs=total_epochs,
        imgsz=imgsz,
        batch=_batch,
        cache=True,
        amp=True,
        device=0,
        lr0=lr0,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        weight_decay=0.001,
        patience=20,
        label_smoothing=0.05,
        hsv_h=aug_hsv_h,
        hsv_s=aug_hsv_s,
        hsv_v=aug_hsv_v,
        degrees=aug_degrees,
        translate=aug_translate,
        scale=aug_scale,
        fliplr=aug_fliplr,
        flipud=aug_flipud,
        mosaic=aug_mosaic,
        close_mosaic=10,
        mixup=aug_mixup,
        copy_paste=aug_copy_paste,
        project=str(settings.model_dir / project_id),
        name="main_model",
        verbose=False,
        workers=0,
    )

    if stop_flag['value']:
        try:
            shutil.rmtree(results.save_dir, ignore_errors=True)
        except Exception:
            pass
        shutil.rmtree(dataset_path, ignore_errors=True)
        from celery.exceptions import Ignore
        raise Ignore()

    best_model_path = results.save_dir / "weights" / "best.pt"
    target_path = settings.model_dir / project_id / "main_best.pt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_model_path, target_path)
    shutil.rmtree(dataset_path)

    final_metrics = epoch_history[-1] if epoch_history else {}

    return {
        "status":           "success",
        "model_path":       str(target_path),
        "model_name":       model_name,
        "use_seed_weights": use_seed_weights,
        "metrics":          final_metrics,
        "history":          epoch_history,
        "split":            {"train": n_train, "val": n_val, "test": n_test},
    }
