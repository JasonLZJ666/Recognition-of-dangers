"""
Build a five-class augmented dataset from the original attachment images.

The final recognition task still uses the five signs from the attachment. This
script creates many view/light variants of those signs so the model can learn
camera angle, illumination, blur, scale and background changes without adding
new classes.

Example:
    python model/build_viewpoint_dataset.py --per-class 220
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.constants import DEFAULT_DATASET_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create viewpoint/light augmented five-class dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset_viewpoint")
    parser.add_argument("--per-class", type=int, default=220)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-source-copy", action="store_true", help="Do not copy the clean source image into the output set")
    return parser.parse_args()


def find_source_image(class_dir: Path) -> Path:
    images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No source image found in {class_dir}")
    return images[0]


def random_background(size: int, rng: random.Random) -> Image.Image:
    base_colors = [
        (242, 242, 244),
        (232, 236, 240),
        (218, 222, 225),
        (204, 207, 210),
        (245, 241, 232),
        (226, 229, 216),
    ]
    color = rng.choice(base_colors)
    bg = Image.new("RGB", (size, size), color)
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    patch_count = rng.randint(2, 5)
    for _ in range(patch_count):
        shade = rng.randint(-28, 24)
        alpha = rng.randint(18, 46)
        fill = (255, 255, 255, alpha) if shade > 0 else (0, 0, 0, alpha)
        x0 = rng.randint(-size // 3, size)
        y0 = rng.randint(-size // 3, size)
        w = rng.randint(size // 3, size)
        h = rng.randint(size // 4, size)
        patch = Image.new("RGBA", (w, h), fill)
        patch = patch.rotate(rng.uniform(-24, 24), expand=True, resample=Image.Resampling.BICUBIC)
        overlay.alpha_composite(patch, (x0, y0))
    bg = Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB")
    return bg.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.6)))


def crop_foreground(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    # Simple white-margin trim that works for the clean PDF icons.
    gray = ImageOps.grayscale(rgba)
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()
    if bbox:
        return rgba.crop(bbox)
    return rgba


def make_mask(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image.convert("RGB"))
    mask = Image.eval(gray, lambda value: 0 if value > 248 else 255)
    return mask.filter(ImageFilter.GaussianBlur(radius=0.4))


def transform_sign(source: Image.Image, size: int, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    sign = crop_foreground(source)
    sign = ImageOps.contain(sign, (int(size * rng.uniform(0.56, 0.86)), int(size * rng.uniform(0.56, 0.86))))
    sign = sign.convert("RGBA")

    brightness = rng.uniform(0.72, 1.28)
    contrast = rng.uniform(0.78, 1.32)
    color = rng.uniform(0.86, 1.14)
    sign_rgb = ImageEnhance.Brightness(sign.convert("RGB")).enhance(brightness)
    sign_rgb = ImageEnhance.Contrast(sign_rgb).enhance(contrast)
    sign_rgb = ImageEnhance.Color(sign_rgb).enhance(color)
    sign = Image.merge("RGBA", (*sign_rgb.split(), sign.getchannel("A")))

    angle = rng.uniform(-24, 24)
    sign = sign.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255, 0))

    if rng.random() < 0.55:
        sign = sign.transform(
            sign.size,
            Image.Transform.AFFINE,
            (1, rng.uniform(-0.18, 0.18), 0, rng.uniform(-0.12, 0.12), 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255, 0),
        )

    if rng.random() < 0.35:
        sign = sign.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))

    mask = sign.getchannel("A")
    return sign, mask


def paste_variant(source: Image.Image, size: int, rng: random.Random) -> Image.Image:
    canvas = random_background(size, rng).convert("RGBA")
    sign, mask = transform_sign(source, size, rng)

    shadow = Image.new("RGBA", sign.size, (0, 0, 0, 0))
    shadow_alpha = mask.filter(ImageFilter.GaussianBlur(radius=rng.uniform(3, 9)))
    shadow.putalpha(ImageEnhance.Brightness(shadow_alpha).enhance(rng.uniform(0.18, 0.38)))

    max_x = max(0, size - sign.width)
    max_y = max(0, size - sign.height)
    offset = (
        rng.randint(0, max_x) if max_x else 0,
        rng.randint(0, max_y) if max_y else 0,
    )
    shadow_offset = (offset[0] + rng.randint(-5, 8), offset[1] + rng.randint(3, 12))
    canvas.alpha_composite(shadow, shadow_offset)
    canvas.alpha_composite(sign, offset)

    out = canvas.convert("RGB")
    if rng.random() < 0.28:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    return out


def build_dataset(source: Path, out: Path, per_class: int, size: int, seed: int, copy_source: bool = True) -> None:
    rng = random.Random(seed)
    if out.exists():
        for file in out.rglob("*"):
            if file.is_file():
                file.unlink()
    out.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise RuntimeError(f"Need at least two class folders under {source}")

    for class_dir in class_dirs:
        class_out = out / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)
        source_image = Image.open(find_source_image(class_dir)).convert("RGB")
        if copy_source:
            source_image.save(class_out / f"{class_dir.name}_source.png")
        for index in range(1, per_class + 1):
            variant = paste_variant(source_image, size, rng)
            variant.save(class_out / f"{class_dir.name}_view_{index:04d}.png", optimize=True)


def main() -> None:
    args = parse_args()
    build_dataset(args.source, args.out, args.per_class, args.size, args.seed, copy_source=not args.no_source_copy)
    total = len(list(args.out.rglob("*.png")))
    print(f"created: {args.out}")
    print(f"classes={len([p for p in args.out.iterdir() if p.is_dir()])} images={total}")


if __name__ == "__main__":
    main()
