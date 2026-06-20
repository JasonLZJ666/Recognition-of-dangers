"""
Evaluate a saved PyTorch checkpoint and export a classification report.

Example:
    python model/evaluate_model.py --checkpoint model/artifacts/best_danger_sign_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.architectures import build_model  # noqa: E402
from model.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_DIR  # noqa: E402
from model.datasets import AugmentedSignDataset  # noqa: E402
from model.model_utils import count_parameters, estimate_model_size_mb  # noqa: E402
from model.reporting import write_json  # noqa: E402
from model.training_engine import run_epoch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a danger-sign checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_ARTIFACT_DIR / "best_danger_sign_model.pt")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_DIR / "classification_report.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    train_config = metadata.get("training_config") or metadata.get("args", {})
    classes = metadata["classes"]
    image_size = int(args.image_size or metadata.get("image_size") or train_config.get("image_size") or 224)
    dropout = float(train_config.get("dropout", 0.35))

    dataset = AugmentedSignDataset(
        args.dataset,
        image_size=image_size,
        repeats=args.repeats,
        training=False,
        auto_augment=False,
    )
    if list(dataset.classes) != list(classes):
        raise RuntimeError(
            "Dataset classes do not match checkpoint metadata: "
            f"dataset={dataset.classes}, checkpoint={classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = build_model(
        arch=metadata["arch"],
        num_classes=len(classes),
        pretrained=False,
        dropout=dropout,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    result = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        classes=classes,
        optimizer=None,
        scaler=None,
        amp=False,
    )
    parameter_summary = count_parameters(model)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "device": str(device),
        "parameter_summary": parameter_summary.to_dict(),
        "estimated_model_size_mb": round(estimate_model_size_mb(model), 3),
        "summary": result.summary.to_dict(),
    }
    write_json(args.out, report)
    print(f"classification report: {args.out}")
    print(f"top1={result.top1:.4f} top3={result.top3:.4f} macro_f1={result.macro_f1:.4f}")


if __name__ == "__main__":
    main()
