"""
Create realistic-looking frontend test photos for the five danger signs.

These images are not new training classes. They are generated from the five
attachment signs and are meant for manual browser upload testing.

Example:
    python model/build_frontend_test_photos.py --per-class 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.constants import DEFAULT_DATASET_DIR, IMAGE_EXTENSIONS, PROJECT_ROOT  # noqa: E402


PHOTO_SIZE = (1280, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frontend upload test photos")
    parser.add_argument("--source", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "test_inputs" / "frontend_photos")
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=6202026)
    return parser.parse_args()


def source_image(class_dir: Path) -> Path:
    candidates = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not candidates:
        raise RuntimeError(f"No source image in {class_dir}")
    return candidates[0]


def trim_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    gray = ImageOps.grayscale(rgba)
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()
    return rgba.crop(bbox) if bbox else rgba


def wall_background(rng: random.Random, size: tuple[int, int]) -> Image.Image:
    width, height = size
    base = rng.choice([(222, 225, 226), (236, 235, 231), (210, 215, 218), (230, 224, 214)])
    image = Image.new("RGB", size, base)
    draw = ImageDraw.Draw(image)

    if rng.random() < 0.65:
        gap = rng.randint(90, 170)
        line_color = tuple(max(0, channel - rng.randint(12, 26)) for channel in base)
        for y in range(rng.randint(-40, 30), height, gap):
            draw.line((0, y, width, y + rng.randint(-8, 8)), fill=line_color, width=rng.randint(2, 4))

    for _ in range(rng.randint(8, 16)):
        x = rng.randint(0, width)
        y = rng.randint(0, height)
        radius = rng.randint(30, 140)
        fill = (
            max(0, min(255, base[0] + rng.randint(-20, 22))),
            max(0, min(255, base[1] + rng.randint(-20, 22))),
            max(0, min(255, base[2] + rng.randint(-20, 22))),
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))


def outdoor_background(rng: random.Random, size: tuple[int, int]) -> Image.Image:
    width, height = size
    sky = rng.choice([(188, 205, 218), (205, 213, 218), (178, 194, 205)])
    ground = rng.choice([(112, 118, 105), (132, 128, 115), (92, 101, 96)])
    image = Image.new("RGB", size, sky)
    draw = ImageDraw.Draw(image)
    horizon = rng.randint(height // 2, int(height * 0.68))
    draw.rectangle((0, horizon, width, height), fill=ground)

    for _ in range(rng.randint(20, 38)):
        x = rng.randint(0, width)
        y = rng.randint(horizon - 70, height)
        color = tuple(max(0, min(255, c + rng.randint(-22, 28))) for c in ground)
        draw.line((x, y, x + rng.randint(-25, 25), y - rng.randint(20, 90)), fill=color, width=rng.randint(2, 5))

    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.4)))


def lab_background(rng: random.Random, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, rng.choice([(240, 241, 243), (232, 234, 236), (225, 229, 232)]))
    draw = ImageDraw.Draw(image)
    table_y = rng.randint(int(height * 0.58), int(height * 0.74))
    draw.rectangle((0, table_y, width, height), fill=rng.choice([(196, 199, 202), (185, 188, 190), (202, 198, 190)]))
    for _ in range(rng.randint(4, 8)):
        x0 = rng.randint(0, width - 160)
        y0 = rng.randint(table_y - 120, table_y + 40)
        draw.rounded_rectangle(
            (x0, y0, x0 + rng.randint(80, 220), y0 + rng.randint(30, 90)),
            radius=8,
            fill=rng.choice([(210, 213, 217), (178, 184, 190), (235, 235, 230)]),
        )
    return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.7, 1.6)))


def make_background(rng: random.Random) -> Image.Image:
    style = rng.choice([wall_background, wall_background, outdoor_background, lab_background])
    background = style(rng, PHOTO_SIZE)
    brightness = ImageEnhance.Brightness(background).enhance(rng.uniform(0.82, 1.18))
    contrast = ImageEnhance.Contrast(brightness).enhance(rng.uniform(0.88, 1.12))
    return contrast.convert("RGBA")


def prepare_sign(image: Image.Image, rng: random.Random) -> Image.Image:
    sign = trim_white(image)
    target = rng.randint(190, 330)
    sign = ImageOps.contain(sign, (target, target)).convert("RGBA")

    rgb = sign.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(rng.uniform(0.78, 1.24))
    rgb = ImageEnhance.Contrast(rgb).enhance(rng.uniform(0.86, 1.26))
    sign = Image.merge("RGBA", (*rgb.split(), sign.getchannel("A")))

    angle = rng.uniform(-22, 22)
    sign = sign.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255, 0))

    if rng.random() < 0.4:
        sign = sign.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75)))

    return sign


def composite_photo(source: Image.Image, rng: random.Random) -> Image.Image:
    canvas = make_background(rng)
    sign = prepare_sign(source, rng)
    alpha = sign.getchannel("A")

    shadow = Image.new("RGBA", sign.size, (0, 0, 0, 0))
    shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(radius=rng.uniform(5, 12)))
    shadow.putalpha(ImageEnhance.Brightness(shadow_alpha).enhance(rng.uniform(0.18, 0.35)))

    max_x = PHOTO_SIZE[0] - sign.width - 80
    max_y = PHOTO_SIZE[1] - sign.height - 70
    x = rng.randint(80, max(80, max_x))
    y = rng.randint(70, max(70, max_y))

    canvas.alpha_composite(shadow, (x + rng.randint(6, 16), y + rng.randint(8, 20)))
    canvas.alpha_composite(sign, (x, y))

    photo = canvas.convert("RGB")
    if rng.random() < 0.35:
        photo = photo.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.45)))
    return photo


def build_photos(source: Path, out: Path, per_class: int, seed: int) -> None:
    rng = random.Random(seed)
    if out.exists():
        for file in out.rglob("*"):
            if file.is_file():
                file.unlink()
    out.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        class_out = out / class_dir.name
        class_out.mkdir(parents=True, exist_ok=True)
        image = Image.open(source_image(class_dir)).convert("RGB")
        for index in range(1, per_class + 1):
            photo = composite_photo(image, rng)
            photo.save(class_out / f"{class_dir.name}_photo_{index:02d}.jpg", quality=92, optimize=True)


def build_contact_sheet(out: Path) -> Path:
    class_dirs = sorted(path for path in out.iterdir() if path.is_dir())
    images_by_class = {
        class_dir.name: sorted(class_dir.glob("*.jpg"))[:8]
        for class_dir in class_dirs
    }
    if not images_by_class:
        raise RuntimeError(f"No generated photos in {out}")

    cell_w, cell_h = 240, 168
    thumb_w, thumb_h = 220, 124
    margin = 18
    columns = len(images_by_class)
    rows = max(len(paths) for paths in images_by_class.values())
    sheet = Image.new(
        "RGB",
        (margin * 2 + columns * cell_w, margin * 2 + rows * cell_h + 24),
        (246, 247, 248),
    )
    draw = ImageDraw.Draw(sheet)

    for col, (class_name, paths) in enumerate(images_by_class.items()):
        x0 = margin + col * cell_w
        draw.text((x0 + 6, margin - 2), class_name, fill=(36, 42, 50))
        for row, path in enumerate(paths):
            y0 = margin + 24 + row * cell_h
            frame = Image.new("RGB", (thumb_w, thumb_h), (255, 255, 255))
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            px = (thumb_w - image.width) // 2
            py = (thumb_h - image.height) // 2
            frame.paste(image, (px, py))
            sheet.paste(frame, (x0 + 6, y0 + 6))
            draw.rectangle(
                (x0 + 6, y0 + 6, x0 + 6 + thumb_w, y0 + 6 + thumb_h),
                outline=(226, 229, 232),
                width=1,
            )
            draw.text((x0 + 6, y0 + thumb_h + 12), path.stem, fill=(76, 82, 90))

    contact_sheet = out / "_contact_sheet.jpg"
    sheet.save(contact_sheet, quality=90, optimize=True)
    return contact_sheet


def main() -> None:
    args = parse_args()
    build_photos(args.source, args.out, args.per_class, args.seed)
    contact_sheet = build_contact_sheet(args.out)
    count = len([path for path in args.out.rglob("*.jpg") if path.name != contact_sheet.name])
    print(f"created: {args.out}")
    print(f"images={count}")
    print(f"contact_sheet={contact_sheet}")


if __name__ == "__main__":
    main()
