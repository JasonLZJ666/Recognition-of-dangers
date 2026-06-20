from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_DIR, IMAGE_EXTENSIONS  # noqa: E402


@dataclass(frozen=True)
class ImageAuditRow:
    class_id: str
    path: str
    width: int
    height: int
    mode: str
    file_size: int
    sha1: str


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_dir: Path) -> dict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    rows: list[ImageAuditRow] = []
    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
            rows.append(
                ImageAuditRow(
                    class_id=class_dir.name,
                    path=str(path.relative_to(dataset_dir)),
                    width=width,
                    height=height,
                    mode=mode,
                    file_size=path.stat().st_size,
                    sha1=sha1_file(path),
                )
            )

    if not rows:
        raise RuntimeError(f"No image files found under {dataset_dir}")

    by_class: dict[str, list[ImageAuditRow]] = defaultdict(list)
    by_hash: dict[str, list[ImageAuditRow]] = defaultdict(list)
    for row in rows:
        by_class[row.class_id].append(row)
        by_hash[row.sha1].append(row)

    width_values = [row.width for row in rows]
    height_values = [row.height for row in rows]
    size_values = [row.file_size for row in rows]
    duplicate_groups = [
        [asdict(row) for row in group]
        for group in by_hash.values()
        if len({row.path for row in group}) > 1
    ]

    return {
        "dataset_dir": str(dataset_dir),
        "class_count": len(by_class),
        "image_count": len(rows),
        "classes": {
            class_id: {
                "image_count": len(items),
                "min_width": min(row.width for row in items),
                "max_width": max(row.width for row in items),
                "min_height": min(row.height for row in items),
                "max_height": max(row.height for row in items),
            }
            for class_id, items in by_class.items()
        },
        "global": {
            "min_width": min(width_values),
            "max_width": max(width_values),
            "min_height": min(height_values),
            "max_height": max(height_values),
            "min_file_size": min(size_values),
            "max_file_size": max(size_values),
            "duplicate_group_count": len(duplicate_groups),
        },
        "duplicates": duplicate_groups,
        "images": [asdict(row) for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit image dataset for danger-sign recognition")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_DIR / "dataset_audit.json")
    args = parser.parse_args()

    report = audit_dataset(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dataset audit: {args.out}")
    print(f"classes={report['class_count']} images={report['image_count']}")


if __name__ == "__main__":
    main()
