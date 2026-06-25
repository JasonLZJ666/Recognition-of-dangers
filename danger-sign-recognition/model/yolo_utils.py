from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    pass


def load_yolo_model(path: Path):
    """Load a YOLO model. Returns None if path missing or ultralytics not installed."""
    if not path.exists():
        return None
    try:
        from ultralytics import YOLO

        return YOLO(str(path))
    except ImportError:
        return None


def yolo_detect_sign(
    model,
    image: Image.Image,
    conf: float = 0.25,
) -> tuple[int, int, int, int] | None:
    """Run YOLO detection, return (x1, y1, x2, y2) of highest-confidence box or None."""
    results = model.predict(image, verbose=False, conf=conf)
    if not results or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    return (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
