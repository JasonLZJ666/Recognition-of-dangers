"""Build a frontend-aligned training dataset from the five source signs.

The original task still has only five classes. This builder creates harder
camera-like scenes from the attachment signs, then crops them with the same
signal-location logic used by the browser app. Training on these crops reduces
the gap between clean synthetic signs and real upload/camera photos.

Example:
    python model/build_scene_crop_dataset.py --per-class 180
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.build_frontend_test_photos import composite_photo, source_image  # noqa: E402
from model.constants import DEFAULT_DATASET_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create scene-crop training dataset for frontend inference")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--base", type=Path, default=PROJECT_ROOT / "dataset_viewpoint")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset_scene_crops")
    parser.add_argument("--per-class", type=int, default=180)
    parser.add_argument(
        "--base-limit-per-class",
        type=int,
        default=None,
        help="Limit copied base/viewpoint images per class; omit to copy all base images",
    )
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=6202126)
    parser.add_argument("--keep-scenes", action="store_true", help="Also save the full generated photos for inspection")
    return parser.parse_args()


def reset_output(out: Path) -> None:
    if out.exists():
        for file in out.rglob("*"):
            if file.is_file():
                file.unlink()
        for folder in sorted((path for path in out.rglob("*") if path.is_dir()), reverse=True):
            folder.rmdir()
    out.mkdir(parents=True, exist_ok=True)


def class_dirs(root: Path) -> list[Path]:
    dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(dirs) < 2:
        raise RuntimeError(f"Need at least two class folders under {root}")
    return dirs


def image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_base_dataset(base: Path, out: Path, limit_per_class: int | None = None) -> int:
    if not base.exists():
        return 0
    copied = 0
    for class_dir in class_dirs(base):
        target_dir = out / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        paths = image_paths(class_dir)
        if limit_per_class is not None:
            paths = paths[: max(0, limit_per_class)]
        for index, path in enumerate(paths, start=1):
            suffix = path.suffix.lower() if path.suffix else ".png"
            shutil.copy2(path, target_dir / f"{class_dir.name}_base_{index:04d}{suffix}")
            copied += 1
    return copied


def browser_signal_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    max_scan = 300
    scale = min(1.0, max_scan / max(width, height))
    scan_width = max(1, round(width * scale))
    scan_height = max(1, round(height * scale))
    scan = image.convert("RGBA").resize((scan_width, scan_height), Image.Resampling.BILINEAR)
    data = np.asarray(scan)
    r = data[:, :, 0].astype(np.int16)
    g = data[:, :, 1].astype(np.int16)
    b = data[:, :, 2].astype(np.int16)
    alpha = data[:, :, 3]

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


def postprocess_crop(crop: Image.Image, size: int, rng: random.Random) -> Image.Image:
    image = ImageOps.exif_transpose(crop).convert("RGB")
    if rng.random() < 0.5:
        image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.75, 1.35))
    if rng.random() < 0.35:
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.88, 1.14))
    if rng.random() < 0.28:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.08, 0.45)))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def build_dataset(
    *,
    source: Path,
    base: Path,
    out: Path,
    per_class: int,
    size: int,
    padding: float,
    seed: int,
    keep_scenes: bool,
    base_limit_per_class: int | None,
) -> dict[str, int]:
    rng = random.Random(seed)
    reset_output(out)
    copied = copy_base_dataset(base, out, base_limit_per_class)
    scene_root = out / "_full_scenes"
    if keep_scenes:
        scene_root.mkdir(parents=True, exist_ok=True)

    generated = 0
    for class_dir in class_dirs(source):
        target_dir = out / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        if keep_scenes:
            (scene_root / class_dir.name).mkdir(parents=True, exist_ok=True)
        source_img = Image.open(source_image(class_dir)).convert("RGB")
        for index in range(1, per_class + 1):
            scene = composite_photo(source_img, rng)
            bounds = square_bounds(browser_signal_bounds(scene), scene.size, padding)
            crop = postprocess_crop(scene.crop(bounds), size, rng)
            crop.save(target_dir / f"{class_dir.name}_scene_crop_{index:04d}.jpg", quality=92, optimize=True)
            if keep_scenes:
                scene.save(scene_root / class_dir.name / f"{class_dir.name}_scene_{index:04d}.jpg", quality=90, optimize=True)
            generated += 1

    return {
        "base_images": copied,
        "scene_crops": generated,
        "total_images": copied + generated,
        "classes": len(class_dirs(source)),
    }


def main() -> None:
    args = parse_args()
    stats = build_dataset(
        source=args.source,
        base=args.base,
        out=args.out,
        per_class=args.per_class,
        size=args.size,
        padding=args.padding,
        seed=args.seed,
        keep_scenes=args.keep_scenes,
        base_limit_per_class=args.base_limit_per_class,
    )
    print(f"created: {args.out}")
    print(
        "classes={classes} base_images={base_images} scene_crops={scene_crops} total_images={total_images}".format(
            **stats
        )
    )


if __name__ == "__main__":
    main()
