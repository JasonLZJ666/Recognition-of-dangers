from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_data(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def save_history_csv(path: Path, history: Iterable[Any]) -> None:
    rows = [to_plain_data(row) for row in history]
    if not rows:
        raise ValueError("history is empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_experiment_report(
    *,
    metadata: dict,
    history: Iterable[Any],
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
    validation_summary: dict | None,
) -> dict:
    history_rows = [to_plain_data(row) for row in history]
    final_row = history_rows[-1] if history_rows else {}
    return {
        "project": "danger-sign-recognition",
        "best_metric": {
            "name": best_metric_name,
            "value": best_metric_value,
            "epoch": best_epoch,
        },
        "final_epoch": final_row,
        "metadata": metadata,
        "validation_summary": validation_summary,
        "history": history_rows,
    }


def write_model_card(
    path: Path,
    *,
    metadata: dict,
    best_metric_name: str,
    best_metric_value: float,
    best_epoch: int,
    validation_summary: dict | None,
) -> None:
    classes = metadata.get("classes", [])
    parameter_summary = metadata.get("parameter_summary", {})
    environment = metadata.get("environment", {})
    lines = [
        "# Danger Sign Recognition Model Card",
        "",
        "## Task",
        "Classify five categories of warning and danger signs for a browser-based safety inspection demo.",
        "",
        "## Model",
        f"- Architecture: `{metadata.get('arch')}`",
        f"- Image size: `{metadata.get('image_size')}`",
        f"- Pretrained: `{metadata.get('pretrained')}`",
        f"- Freeze backbone: `{metadata.get('freeze_backbone')}`",
        f"- Total parameters: `{parameter_summary.get('total')}`",
        f"- Trainable parameters: `{parameter_summary.get('trainable')}`",
        f"- Estimated model size MB: `{metadata.get('estimated_model_size_mb')}`",
        "",
        "## Training",
        f"- Best metric: `{best_metric_name}` = `{best_metric_value:.6f}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Device: `{environment.get('device')}`",
        f"- PyTorch: `{environment.get('torch')}`",
        "",
        "## Classes",
    ]
    lines.extend(f"- `{class_id}`" for class_id in classes)
    if validation_summary:
        lines.extend(
            [
                "",
                "## Validation Summary",
                f"- Loss: `{validation_summary.get('loss')}`",
                f"- Top-1: `{validation_summary.get('top1')}`",
                f"- Top-3: `{validation_summary.get('top3')}`",
                f"- Macro F1: `{validation_summary.get('macro_f1')}`",
                f"- Weighted F1: `{validation_summary.get('weighted_f1')}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
