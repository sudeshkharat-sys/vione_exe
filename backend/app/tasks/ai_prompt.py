import torch
from PIL import Image
import numpy as np
import uuid
import json
from pathlib import Path
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection, 
    Sam2Model, 
    Sam2Processor,
    AutoModelForZeroShotImageClassification
)
import supervision as sv

from .celery_app import celery_app
from ..config import settings
from ..connectors.statedb_connector import StateDBConnector

# ── Model Cache ───────────────────────────────────────────────────────
_MODELS = {
    "p_dino": None, "m_dino": None,
    "p_sam2": None, "m_sam2": None,
    "p_siglip": None, "m_siglip": None
}

def load_ai_models():
    global _MODELS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if _MODELS["m_dino"] is None:
        print(f"Loading AI Prompt models to {device}...")

        # Use paths from settings (set via env vars in docker-compose / .env).
        # Do NOT compute paths relative to __file__ — the Docker working directory
        # makes parents[3] resolve to "/" causing HuggingFace repo-ID validation errors.
        dino_path   = settings.grounding_dino_path
        sam2_path   = settings.sam2_path
        siglip_path = settings.siglip_path

        # Validate paths exist before calling from_pretrained. HuggingFace falls back
        # to repo-ID validation when the directory is missing, producing a misleading
        # "Repo id must be in the form..." error instead of a clear "not found" message.
        for name, path in [("GROUNDING_DINO_PATH", dino_path), ("SAM2_PATH", sam2_path), ("SIGLIP_PATH", siglip_path)]:
            if not Path(path).is_dir():
                raise FileNotFoundError(
                    f"Model directory not found: {path}\n"
                    f"Set the {name} env var to the correct local path, or copy the model files there.\n"
                    f"Expected inside the container at: {path}"
                )

        print(f"  - Loading DINO from {dino_path}...")
        _MODELS["p_dino"] = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
        _MODELS["m_dino"] = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to(device)

        print(f"  - Loading SAM 2 from {sam2_path}...")
        _MODELS["p_sam2"] = Sam2Processor.from_pretrained(sam2_path, local_files_only=True)
        _MODELS["m_sam2"] = Sam2Model.from_pretrained(sam2_path, local_files_only=True).to(device)

        print(f"  - Loading SigLIP from {siglip_path}...")
        _MODELS["p_siglip"] = AutoProcessor.from_pretrained(siglip_path, local_files_only=True)
        _MODELS["m_siglip"] = AutoModelForZeroShotImageClassification.from_pretrained(siglip_path, local_files_only=True).to(device)
    
    return _MODELS, device

# ── Helpers ───────────────────────────────────────────────────────────

def verify_with_siglip(image, boxes, text_prompt, processor, model, device, threshold=0.7):
    if len(boxes) == 0: return []
    verified_indices = []
    candidate_labels = [f"a photo of {text_prompt}", "background", "something else"]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        x1_p, y1_p = max(0, x1 - int(w*0.2)), max(0, y1 - int(h*0.2))
        x2_p, y2_p = min(image.width, x2 + int(w*0.2)), min(image.height, y2 + int(h*0.2))
        
        crop = image.crop((x1_p, y1_p, x2_p, y2_p))
        inputs = processor(text=candidate_labels, images=crop, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
        if probs[0] > threshold:
            verified_indices.append(i)
    return verified_indices

def refine_masks(image, boxes, processor, model, device):
    width, height = image.size
    inputs = processor(image, input_boxes=[boxes.tolist()], return_tensors="pt")
    inputs_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs_gpu, multimask_output=True)
    
    final_masks = []
    for i in range(len(boxes)):
        iou_scores = outputs.iou_scores[0, i].cpu().numpy()
        best_idx = np.argmax(iou_scores)
        raw_mask = outputs.pred_masks[0, i, best_idx]
        mask_full = torch.nn.functional.interpolate(
            raw_mask.unsqueeze(0).unsqueeze(0), 
            size=(height, width), 
            mode="bilinear", 
            align_corners=False
        )[0, 0]
        final_masks.append((mask_full > 0.0).cpu().numpy().astype(bool))
    return np.array(final_masks)

# ── Tasks ─────────────────────────────────────────────────────────────

@celery_app.task(name="app.tasks.ai_prompt.detect_with_prompt", bind=True)
def detect_with_prompt(self, project_id: str, image_id: str, prompt: str, clear_existing: bool = False):
    db = StateDBConnector()
    models, device = load_ai_models()
    
    # 1. Fetch image path
    with db.get_session() as conn:
        img_rows = db.execute_query(conn, "SELECT id, filepath FROM images WHERE id = :id", {"id": image_id})
        if not img_rows: return {"error": "Image not found"}
        img_row = img_rows[0]

    real_path = Path(".") / img_row["filepath"].lstrip("/")
    if not real_path.exists():
        real_path = settings.upload_dir.parent / Path(img_row["filepath"].lstrip("/"))
    
    if not real_path.exists(): return {"error": f"File not found: {real_path}"}

    image = Image.open(real_path).convert("RGB")
    w, h = image.size

    # 2. Stage 1: Discovery (DINO)
    discovery_prompt = f"{prompt} . object ."
    inputs_dino = models["p_dino"](images=image, text=discovery_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_dino = models["m_dino"](**inputs_dino)
    
    results_dino = models["p_dino"].post_process_grounded_object_detection(
        outputs_dino, inputs_dino.input_ids,
        threshold=settings.dino_box_threshold,
        text_threshold=settings.dino_text_threshold,
        target_sizes=torch.Tensor([[h, w]]).to(device)
    )[0]
    
    boxes = results_dino["boxes"].cpu().numpy()
    scores = results_dino["scores"].cpu().numpy()
    
    if len(boxes) == 0: return {"status": "success", "count": 0}

    # 3. NMS & Size Filter
    predictions = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)
    keep = sv.box_non_max_suppression(predictions, iou_threshold=0.5)
    boxes = boxes[keep]
    scores = scores[keep]

    img_area = w * h
    size_keep = [i for i, b in enumerate(boxes) if ((b[2]-b[0])*(b[3]-b[1])) < (img_area * 0.4)]
    boxes = boxes[size_keep]
    scores = scores[size_keep]

    if len(boxes) == 0: return {"status": "success", "count": 0}

    # 4. Stage 2: Verification (SigLIP)
    verified_idx = verify_with_siglip(image, boxes, prompt, models["p_siglip"], models["m_siglip"], device)
    if not verified_idx: return {"status": "success", "count": 0}
    verified_boxes = boxes[verified_idx]
    verified_scores = scores[verified_idx]

    # 4b. Post-verification NMS — remove overlapping boxes that survived SigLIP
    if len(verified_boxes) > 1:
        v_preds = np.concatenate([verified_boxes, verified_scores.reshape(-1, 1)], axis=1)
        v_keep = sv.box_non_max_suppression(v_preds, iou_threshold=0.4)
        verified_boxes = verified_boxes[v_keep]

    # 5. Stage 3: Segmentation (SAM 2)
    masks = refine_masks(image, verified_boxes, models["p_sam2"], models["m_sam2"], device)

    # 6. Save to DB
    with db.get_session() as conn:
        # ── 6a. Save annotations with empty class (user will classify via UI) ──
        if clear_existing:
            db.execute_update(conn, "DELETE FROM annotations WHERE image_id = :id", {"id": image_id})

        ann_rows = []
        for i in range(len(verified_boxes)):
            # Convert box to normalized xywh for platform consistency
            box = verified_boxes[i]
            # detection_anything uses absolute xyxy
            # Platform uses normalized xywh: [center_x, center_y, width, height]
            nx = ((box[0] + box[2]) / 2) / w
            ny = ((box[1] + box[3]) / 2) / h
            nw = (box[2] - box[0]) / w
            nh = (box[3] - box[1]) / h

            ann_rows.append({
                "id": str(uuid.uuid4()),
                "image_id": image_id,
                "class_name": "",
                "bbox": json.dumps([float(nx), float(ny), float(nw), float(nh)]),
                "source": "ai_prompt"
            })

        db.execute_many(
            conn,
            "INSERT INTO annotations (id, image_id, class_name, bbox, source) "
            "VALUES (:id, :image_id, :class_name, CAST(:bbox AS JSONB), :source)",
            ann_rows
        )
        db.execute_update(conn, "UPDATE images SET status = 'annotated' WHERE id = :id", {"id": image_id})

    return {"status": "success", "count": len(ann_rows)}

@celery_app.task(name="app.tasks.ai_prompt.bulk_detect_with_prompt", bind=True)
def bulk_detect_with_prompt(self, project_id: str, prompt: str, image_ids: list = None):
    db = StateDBConnector()
    
    if not image_ids:
        with db.get_session() as conn:
            rows = db.execute_query(
                conn, 
                "SELECT id FROM images WHERE project_id = :pid AND status = 'pending'",
                {"pid": project_id}
            )
            image_ids = [r["id"] for r in rows]
    
    if not image_ids: return {"status": "success", "count": 0}

    total = len(image_ids)
    processed = 0
    total_found = 0

    for i, img_id in enumerate(image_ids):
        self.update_state(state="STARTED", meta={"current": i+1, "total": total})
        res = detect_with_prompt(project_id, img_id, prompt, clear_existing=False)
        if "count" in res:
            total_found += res["count"]
        processed += 1
    
    return {"status": "success", "processed": processed, "total_found": total_found}
