#!/usr/bin/env python3
"""
Pre-download YOLO model weights for offline deployment.

Run during Docker image build so all YOLO base weights are embedded in the
image and available without any internet access at runtime.

Usage:
    python scripts/download_yolo_weights.py
    YOLO_WEIGHTS_DIR=/custom/path python scripts/download_yolo_weights.py
"""
import os
import sys
from pathlib import Path

WEIGHTS_DIR = Path(os.environ.get("YOLO_WEIGHTS_DIR", "/app/data/yolo_weights"))

# Matches every entry in YOLO_MODELS from backend/app/api/pipeline.py.
# Ordered newest-first so the most-requested models are downloaded first.
ALL_MODELS = [
    # YOLO26 (latest Ultralytics — NMS-free)
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    # YOLO12 (attention-centric)
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
    # YOLO11
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # YOLOv10
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolov10b.pt", "yolov10l.pt", "yolov10x.pt",
    # YOLOv9
    "yolov9c.pt", "yolov9e.pt",
    # YOLOv8
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
]

_MIN_SIZE_BYTES = 1024 * 1024  # 1 MB — anything smaller is a corrupt/partial download


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1_048_576:.1f} MB"


def download_weights(weights_dir: Path = WEIGHTS_DIR) -> list[str]:
    """
    Download every model in ALL_MODELS into *weights_dir*.

    ultralytics resolves unqualified names (e.g. "yolov8n.pt") by checking
    the current working directory first, then downloading from GitHub.
    Changing CWD to weights_dir before calling YOLO() causes the file to land
    directly there without any extra copy step.

    Returns a list of model names that could not be downloaded.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(weights_dir)

    try:
        # Import after chdir so ultralytics picks up the right settings path
        from ultralytics import YOLO  # noqa: PLC0415

        failed: list[str] = []

        for model_name in ALL_MODELS:
            dest = weights_dir / model_name

            if dest.exists() and dest.stat().st_size >= _MIN_SIZE_BYTES:
                print(f"  SKIP   {model_name:<22}  already present  ({_mb(dest)})",
                      flush=True)
                continue

            print(f"  DOWN   {model_name:<22}  downloading …", flush=True)
            try:
                YOLO(model_name)
                if dest.exists() and dest.stat().st_size >= _MIN_SIZE_BYTES:
                    print(f"  OK     {model_name:<22}  ({_mb(dest)})", flush=True)
                else:
                    # Ultralytics may have cached it in ~/.config/Ultralytics/
                    # instead of CWD for some model families.  Try to find and
                    # copy the cached file.
                    _copy_from_ultralytics_cache(model_name, dest)
                    if dest.exists():
                        print(f"  OK     {model_name:<22}  (copied from cache, {_mb(dest)})",
                              flush=True)
                    else:
                        print(f"  WARN   {model_name:<22}  loaded but file not in {weights_dir}",
                              flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL   {model_name:<22}  {exc}", flush=True)
                failed.append(model_name)

        return failed

    finally:
        os.chdir(original_cwd)


def _copy_from_ultralytics_cache(model_name: str, dest: Path) -> None:
    """Best-effort: search common ultralytics cache directories for *model_name*."""
    import shutil

    candidates: list[Path] = []

    # ultralytics ≥ 8.1 stores assets under SETTINGS['weights_dir']
    try:
        from ultralytics.utils import SETTINGS
        w = Path(SETTINGS.get("weights_dir", ""))
        if w.is_dir():
            candidates.append(w / model_name)
    except Exception:  # noqa: BLE001
        pass

    # Common fallback locations
    home = Path.home()
    candidates += [
        home / ".config" / "Ultralytics" / model_name,
        home / ".cache"  / "ultralytics" / model_name,
        Path("/tmp") / model_name,
    ]

    for src in candidates:
        if src.exists() and src.stat().st_size >= _MIN_SIZE_BYTES:
            shutil.copy2(src, dest)
            return


def main() -> int:
    print(f"\n{'='*60}")
    print(f"  YOLO weight pre-download")
    print(f"  Target directory : {WEIGHTS_DIR}")
    print(f"  Models           : {len(ALL_MODELS)}")
    print(f"{'='*60}\n")

    failed = download_weights()

    print(f"\n{'='*60}")
    ok = len(ALL_MODELS) - len(failed)
    print(f"  Done: {ok}/{len(ALL_MODELS)} weights available")
    if failed:
        print(f"  Failed ({len(failed)}): {', '.join(failed)}")
        print("  (Newer model families may not yet have public releases — this is non-fatal)")
    print(f"{'='*60}\n")

    # Non-zero only when ALL downloads fail — partial failures are acceptable
    # because some model families (e.g. YOLO26) may not have public releases yet.
    return 1 if len(failed) == len(ALL_MODELS) else 0


if __name__ == "__main__":
    sys.exit(main())
