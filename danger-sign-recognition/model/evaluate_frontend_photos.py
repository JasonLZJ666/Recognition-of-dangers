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
    parser.add_argument("--padding", type=float, default=0.28)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def image_files(root: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((path, class_dir.name))
    return items


def count_mask_in_box(
    mask: np.ndarray,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    padding: int,
) -> int:
    height, width = mask.shape
    left = max(0, min_x - padding)
    top = max(0, min_y - padding)
    right = min(width - 1, max_x + padding)
    bottom = min(height - 1, max_y + padding)
    return int(mask[top : bottom + 1, left : right + 1].sum())


def largest_component_bounds(
    mask: np.ndarray,
    min_pixels: int,
    support_mask: np.ndarray | None = None,
) -> tuple[int, int, int, int] | None:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: tuple[int, int, int, int, int, float] | None = None

    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_x), int(start_y))]
        visited[start_y, start_x] = True
        count = 0
        min_x = width
        min_y = height
        max_x = 0
        max_y = 0

        while stack:
            x, y = stack.pop()
            count += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))

        if count >= min_pixels:
            area = (max_x - min_x + 1) * (max_y - min_y + 1)
            support = count_mask_in_box(support_mask, min_x, min_y, max_x, max_y, 10) if support_mask is not None else 0
            if support_mask is not None:
                score = support * 50 + count * 0.2 if support > 0 else count * 0.03
            else:
                score = count + float(np.sqrt(area)) * 0.2
            if best is None or score > best[5]:
                best = (min_x, min_y, max_x, max_y, count, score)

    if best is None:
        return None
    min_x, min_y, max_x, max_y, _, _ = best
    return min_x, min_y, max_x, max_y


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
    max_channel = np.maximum.reduce([r, g, b])
    min_channel = np.minimum.reduce([r, g, b])
    saturation = np.divide(
        max_channel - min_channel,
        max_channel,
        out=np.zeros_like(max_channel, dtype=np.float32),
        where=max_channel != 0,
    )
    yellow_balance = (r + g) / 2 - b
    relative_yellow = (r > b + 24) & (g > b + 18) & (yellow_balance > 28)
    bright_yellow = (r > 130) & (g > 105) & (b < 150) & ((r + g - b) > 210)
    shadow_yellow = (r > 70) & (g > 58) & (b < 130) & relative_yellow & (saturation > 0.18)
    highlight_yellow = (r > 190) & (g > 170) & (b < 175) & relative_yellow & (saturation > 0.2)
    yellow_core = (alpha > 20) & (bright_yellow | shadow_yellow | highlight_yellow)
    non_white = (alpha > 20) & ((r < 242) | (g < 242) | (b < 232))
    dark_stroke = (r < 125) & (g < 120) & (b < 115)
    warm_stroke = (r > 155) & (g > 125) & (b < 115)
    sign_color = non_white & (yellow_core | dark_stroke | warm_stroke)

    selected = largest_component_bounds(yellow_core, min_pixels=35, support_mask=dark_stroke)
    if selected is None:
        selected = largest_component_bounds(sign_color, min_pixels=80)
    if selected is None:
        return 0, 0, width, height

    min_x, min_y, max_x, max_y = selected
    pad_x = max(4, round((max_x - min_x) * 0.08))
    pad_y = max(4, round((max_y - min_y) * 0.08))
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
