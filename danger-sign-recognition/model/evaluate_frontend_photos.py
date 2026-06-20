"""Evaluate frontend-style scene photos with the browser crop preprocessing.

This script mirrors the signal-box crop used by app/app.js before running the
PyTorch checkpoint. It helps distinguish model weakness from frontend
preprocessing mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.architectures import build_model  # noqa: E402
from model.constants import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frontend scene photos")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test_inputs" / "frontend_photos")
    parser.add_argument("--crop", choices=["browser", "full", "padded", "square"], default="browser")
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def image_files(root: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, class_dir.name))
    return items


def browser_signal_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    max_scan = 300
    scale = min(1.0, max_scan / max(width, height))
    scan_width = max(1, round(width * scale))
    scan_height = max(1, round(height * scale))
    scan = image.convert("RGBA").resize((scan_width, scan_height), Image.Resampling.BILINEAR)
    data = np.asarray(scan)
    rgb = data[:, :, :3]
    alpha = data[:, :, 3]
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    yellow_core = (alpha > 20) & (r > 130) & (g > 105) & (b < 135) & ((r - b) > 38) & ((g - b) > 25) & ((r + g - b) > 220)
    non_white = (alpha > 20) & ((r < 242) | (g < 242) | (b < 232))
    sign_color = non_white & ((r < 120) | (g < 120) | (b < 120) | ((r > 160) & (g > 135) & (b < 100)))
    yellow_ys, yellow_xs = np.where(yellow_core)
    if len(yellow_xs) >= 45:
        ys, xs = yellow_ys, yellow_xs
    else:
        ys, xs = np.where(sign_color)
    if len(xs) < 80:
        return 0, 0, width, height

    min_x = int(xs.min())
    min_y = int(ys.min())
    max_x = int(xs.max())
    max_y = int(ys.max())
    pad_x = max(4, round((max_x - min_x) * 0.04))
    pad_y = max(4, round((max_y - min_y) * 0.04))
    min_x = max(0, min_x - pad_x)
    min_y = max(0, min_y - pad_y)
    max_x = min(scan_width - 1, max_x + pad_x)
    max_y = min(scan_height - 1, max_y + pad_y)

    return (
        max(0, int(min_x / scale)),
        max(0, int(min_y / scale)),
        min(width, int((max_x + 1) / scale)),
        min(height, int((max_y + 1) / scale)),
    )


def padded_bounds(bounds: tuple[int, int, int, int], size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    width, height = size
    bw = x2 - x1
    bh = y2 - y1
    pad = int(max(bw, bh) * padding)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def square_bounds(bounds: tuple[int, int, int, int], size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bounds
    width, height = size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = min(width, height, max(x2 - x1, y2 - y1) * (1 + padding * 2))
    side = max(1, side)
    left = min(max(0, cx - side / 2), max(0, width - side))
    top = min(max(0, cy - side / 2), max(0, height - side))
    right = left + side
    bottom = top + side
    return (
        max(0, int(round(left))),
        max(0, int(round(top))),
        min(width, int(round(right))),
        min(height, int(round(bottom))),
    )


def preprocess(image: Image.Image, image_size: int, crop_mode: str, padding: float) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if crop_mode != "full":
        bounds = browser_signal_bounds(image)
        if crop_mode == "padded":
            bounds = padded_bounds(bounds, image.size, padding)
        elif crop_mode == "square":
            bounds = square_bounds(bounds, image.size, padding)
        image = image.crop(bounds)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    train_config = metadata.get("training_config") or metadata.get("args", {})
    classes = list(metadata["classes"])
    image_size = int(metadata.get("image_size") or train_config.get("image_size") or 224)
    model = build_model(
        arch=metadata["arch"],
        num_classes=len(classes),
        pretrained=False,
        dropout=float(train_config.get("dropout", 0.35)),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    correct = 0
    total = 0
    per_class: dict[str, Counter] = defaultdict(Counter)
    mistakes: list[str] = []
    with torch.no_grad():
        for path, truth in image_files(args.dataset):
            tensor = preprocess(Image.open(path), image_size, args.crop, args.padding)
            logits = model(tensor)
            pred = classes[int(torch.argmax(logits, dim=1).item())]
            total += 1
            correct += int(pred == truth)
            per_class[truth][pred] += 1
            if pred != truth:
                mistakes.append(f"{truth}/{path.name} -> {pred}")

    print(f"dataset={args.dataset}")
    print(f"crop={args.crop} total={total} correct={correct} accuracy={correct / max(1, total):.4f}")
    for truth in classes:
        row = per_class.get(truth, Counter())
        support = sum(row.values())
        ok = row.get(truth, 0)
        print(f"{truth}: {ok}/{support} {ok / max(1, support):.4f} predictions={dict(row)}")
    if mistakes:
        print("mistakes:")
        for item in mistakes[:30]:
            print(f"  {item}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "crop": args.crop,
            "padding": args.padding,
            "total": total,
            "correct": correct,
            "accuracy": correct / max(1, total),
            "classes": classes,
            "per_class": {
                truth: {
                    "correct": per_class.get(truth, Counter()).get(truth, 0),
                    "support": sum(per_class.get(truth, Counter()).values()),
                    "predictions": dict(per_class.get(truth, Counter())),
                }
                for truth in classes
            },
            "mistakes": mistakes,
        }
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report={args.out}")


if __name__ == "__main__":
    main()
