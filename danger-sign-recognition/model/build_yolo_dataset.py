"""
Generate a YOLO-format object detection dataset for danger sign localization.

Reuses the composite scene generation from build_frontend_test_photos.py, but
records the bounding box of each sign in YOLO annotation format. The dataset
uses a single class (danger_sign, id=0) because classification is handled by
the downstream EfficientNet model.

Example:
    python model/build_yolo_dataset.py --count 1000
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.build_frontend_test_photos import (  # noqa: E402
    PHOTO_SIZE,
    make_background,
    prepare_sign,
    source_image,
    trim_white,
)
from model.constants import DEFAULT_DATASET_DIR, DEFAULT_YOLO_DATASET_DIR, IMAGE_EXTENSIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO detection dataset for danger signs")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_YOLO_DATASET_DIR)
    parser.add_argument("--count", type=int, default=1000, help="Total images to generate")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--multi-sign-ratio", type=float, default=0.1,
                        help="Fraction of images with 2 signs")
    return parser.parse_args()


def sign_tight_bbox(sign: Image.Image) -> tuple[int, int, int, int] | None:
    """Get bounding box of non-transparent pixels in RGBA sign image."""
    alpha = sign.getchannel("A")
    bbox = alpha.getbbox()
    return bbox


def composite_photo_with_bbox(
    source: Image.Image,
    rng: random.Random,
    *,
    min_scale: int = 80,
    max_scale: int = 450,
) -> tuple[Image.Image, list[tuple[int, int, int, int]]]:
    """Composite a sign onto a background, returning image and list of bounding boxes."""
    canvas = make_background(rng)
    sign = trim_white(source.convert("RGBA"))

    target = rng.randint(min_scale, max_scale)
    sign = ImageOps.contain(sign, (target, target)).convert("RGBA")

    rgb = sign.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.72, 1.28))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.82, 1.28))
    sign = Image.merge("RGBA", (*rgb.split(), sign.getchannel("A")))

    angle = rng.uniform(-25, 25)
    sign = sign.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC,
                       fillcolor=(255, 255, 255, 0))

    if rng.random() < 0.4:
        sign = sign.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.8)))

    alpha = sign.getchannel("A")
    shadow = Image.new("RGBA", sign.size, (0, 0, 0, 0))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(5, 12)))
    shadow.putalpha(ImageEnhance.Brightness(shadow_alpha).enhance(rng.uniform(0.18, 0.35)))

    margin = 40
    max_x = PHOTO_SIZE[0] - sign.width - margin
    max_y = PHOTO_SIZE[1] - sign.height - margin
    x = rng.randint(margin, max(margin, max_x))
    y = rng.randint(margin, max(margin, max_y))

    canvas.alpha_composite(shadow, (x + rng.randint(6, 16), y + rng.randint(8, 20)))
    canvas.alpha_composite(sign, (x, y))

    tight = sign_tight_bbox(sign)
    if tight is not None:
        tx1, ty1, tx2, ty2 = tight
        bbox = (x + tx1, y + ty1, x + tx2, y + ty2)
    else:
        bbox = (x, y, x + sign.width, y + sign.height)

    bbox = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(PHOTO_SIZE[0], bbox[2]),
        min(PHOTO_SIZE[1], bbox[3]),
    )

    photo = canvas.convert("RGB")
    if rng.random() < 0.35:
        photo = photo.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.5)))
    return photo, [bbox]


def to_yolo_annotation(
    bboxes: list[tuple[int, int, int, int]],
    img_width: int,
    img_height: int,
    class_id: int = 0,
) -> str:
    """Convert pixel bboxes to YOLO format: class_id cx cy w h (normalized 0-1)."""
    lines = []
    for x1, y1, x2, y2 in bboxes:
        cx = (x1 + x2) / 2.0 / img_width
        cy = (y1 + y2) / 2.0 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def write_dataset_yaml(out: Path) -> Path:
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: danger_sign\n",
        encoding="utf-8",
    )
    return yaml_path


def build_dataset(
    source: Path,
    out: Path,
    count: int,
    val_ratio: float,
    seed: int,
    multi_sign_ratio: float,
) -> None:
    rng = random.Random(seed)

    if out.exists():
        shutil.rmtree(out)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    class_dirs = sorted(p for p in source.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class directories found in {source}")

    sources = {}
    for class_dir in class_dirs:
        src_path = source_image(class_dir)
        sources[class_dir.name] = Image.open(src_path).convert("RGB")

    class_names = list(sources.keys())
    val_count = int(count * val_ratio)
    train_count = count - val_count

    for split, split_count in [("train", train_count), ("val", val_count)]:
        for i in range(split_count):
            cls_name = class_names[i % len(class_names)]
            src_img = sources[cls_name]

            photo, bboxes = composite_photo_with_bbox(src_img, rng)

            if rng.random() < multi_sign_ratio:
                other_cls = rng.choice(class_names)
                other_img = sources[other_cls]
                extra_sign = trim_white(other_img.convert("RGBA"))
                extra_target = rng.randint(60, 200)
                extra_sign = ImageOps.contain(extra_sign, (extra_target, extra_target)).convert("RGBA")
                extra_angle = rng.uniform(-20, 20)
                extra_sign = extra_sign.rotate(extra_angle, expand=True,
                                               resample=Image.Resampling.BICUBIC,
                                               fillcolor=(255, 255, 255, 0))

                margin = 20
                ex = rng.randint(margin, max(margin, PHOTO_SIZE[0] - extra_sign.width - margin))
                ey = rng.randint(margin, max(margin, PHOTO_SIZE[1] - extra_sign.height - margin))

                photo_rgba = photo.convert("RGBA")
                photo_rgba.alpha_composite(extra_sign, (ex, ey))
                photo = photo_rgba.convert("RGB")

                tight = sign_tight_bbox(extra_sign)
                if tight is not None:
                    tx1, ty1, tx2, ty2 = tight
                    extra_bbox = (ex + tx1, ey + ty1, ex + tx2, ey + ty2)
                else:
                    extra_bbox = (ex, ey, ex + extra_sign.width, ey + extra_sign.height)
                extra_bbox = (
                    max(0, extra_bbox[0]),
                    max(0, extra_bbox[1]),
                    min(PHOTO_SIZE[0], extra_bbox[2]),
                    min(PHOTO_SIZE[1], extra_bbox[3]),
                )
                bboxes.append(extra_bbox)

            stem = f"scene_{split}_{i:05d}"
            photo.save(out / "images" / split / f"{stem}.jpg", quality=92, optimize=True)

            annotation = to_yolo_annotation(bboxes, PHOTO_SIZE[0], PHOTO_SIZE[1])
            (out / "labels" / split / f"{stem}.txt").write_text(annotation, encoding="utf-8")

    write_dataset_yaml(out)


def main() -> None:
    args = parse_args()
    print(f"生成 YOLO 数据集: {args.out}")
    print(f"  总数={args.count}  验证比例={args.val_ratio}  多标志比例={args.multi_sign_ratio}")
    build_dataset(args.source, args.out, args.count, args.val_ratio, args.seed, args.multi_sign_ratio)
    train_count = len(list((args.out / "images" / "train").glob("*.jpg")))
    val_count = len(list((args.out / "images" / "val").glob("*.jpg")))
    print(f"完成: train={train_count}  val={val_count}")
    print(f"配置: {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
