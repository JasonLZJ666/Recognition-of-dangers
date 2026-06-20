"""
Export the trained PyTorch checkpoint to an ONNX model for browser inference.

Output files:
  app/model/danger_sign_model.onnx
  app/model/model_metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from model.architectures import build_model  # noqa: E402
from model.constants import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export danger-sign model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "model" / "artifacts" / "best_danger_sign_model.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "app" / "model")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    train_args = metadata.get("training_config") or metadata.get("args", {})
    classes = metadata["classes"]
    image_size = int(metadata["image_size"])

    model = build_model(
        arch=metadata["arch"],
        num_classes=len(classes),
        pretrained=False,
        dropout=float(train_args.get("dropout", 0.35)),
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / "danger_sign_model.onnx"
    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )

    browser_metadata = {
        "model": "danger_sign_model.onnx",
        "externalData": "danger_sign_model.onnx.data",
        "externalDataPath": "danger_sign_model.onnx.data",
        "inputName": "input",
        "outputName": "logits",
        "classes": classes,
        "classToIdx": metadata["class_to_idx"],
        "imageSize": image_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "sourceCheckpoint": str(args.checkpoint),
        "arch": metadata["arch"],
        "bestEpoch": metadata.get("best_epoch"),
        "bestValAcc": metadata.get("best_val_acc"),
        "bestMetricName": metadata.get("best_metric_name"),
        "bestMetricValue": metadata.get("best_metric_value"),
    }
    (args.out_dir / "model_metadata.json").write_text(
        json.dumps(browser_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"exported: {onnx_path}")
    print(f"metadata: {args.out_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
