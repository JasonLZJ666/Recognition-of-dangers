"""
Command-line entry point for training the danger-sign recognition model.

The training implementation is intentionally split into modules:
  - constants.py: shared paths and artifact names
  - datasets.py: dataset discovery and augmentation
  - architectures.py: EfficientNet/ResNet/MobileNet/custom CNN
  - metrics.py: accuracy, F1 and confusion-matrix utilities
  - callbacks.py: early stopping and checkpoint writing
  - reporting.py: CSV/JSON/model-card artifact export
  - data_audit.py: dataset quality audit

Recommended quick start:
    python -m pip install torch torchvision pillow
    python model/train_model.py --arch efficientnet_b0 --pretrained --epochs 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from model.architectures import build_model
    from model.callbacks import CheckpointWriter, EarlyStopping
    from model.config import TrainingConfig
    from model.constants import ARTIFACT_FILENAMES, DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_DIR, SUPPORTED_ARCHITECTURES
    from model.data_audit import audit_dataset
    from model.datasets import AugmentedSignDataset
    from model.model_utils import (
        collect_trainable_layer_names,
        count_parameters,
        describe_optimizer,
        estimate_model_size_mb,
        format_large_int,
        snapshot_environment,
    )
    from model.reporting import build_experiment_report, save_history_csv, write_json, write_model_card
    from model.training_engine import EpochMetrics, choose_best_metric, current_lr, run_epoch, set_seed
except ModuleNotFoundError as exc:  # pragma: no cover - user environment guard
    raise SystemExit(
        "Training dependencies are missing.\n"
        "Install them first: python -m pip install torch torchvision pillow"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train danger-sign recognition model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--arch", choices=SUPPORTED_ARCHITECTURES, default="efficientnet_b0")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for torchvision backbones")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze most feature layers for small datasets")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--metric-for-best", choices=("val_macro_f1", "val_top1", "val_top3"), default="val_macro_f1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-auto-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_loaders(config: TrainingConfig, device: torch.device) -> tuple[AugmentedSignDataset, DataLoader, DataLoader]:
    train_dataset = AugmentedSignDataset(
        config.dataset,
        image_size=config.image_size,
        repeats=config.repeats,
        training=True,
        auto_augment=config.auto_augment,
    )
    val_dataset = AugmentedSignDataset(
        config.dataset,
        image_size=config.image_size,
        repeats=config.repeats,
        training=False,
        auto_augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
    )
    return train_dataset, train_loader, val_loader


def make_metadata(
    *,
    config: TrainingConfig,
    dataset: AugmentedSignDataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataset_audit: dict,
) -> dict:
    parameter_summary = count_parameters(model)
    return {
        "arch": config.arch,
        "pretrained": config.pretrained,
        "freeze_backbone": config.freeze_backbone,
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "image_size": config.image_size,
        "training_config": config.to_json_dict(),
        "parameter_summary": parameter_summary.to_dict(),
        "estimated_model_size_mb": round(estimate_model_size_mb(model), 3),
        "trainable_layers_preview": collect_trainable_layer_names(model),
        "optimizer": describe_optimizer(optimizer),
        "environment": snapshot_environment(device, config.amp),
        "dataset_audit": {
            "class_count": dataset_audit["class_count"],
            "image_count": dataset_audit["image_count"],
            "global": dataset_audit["global"],
        },
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    config = TrainingConfig.from_args(args, amp=amp)
    set_seed(config.seed, deterministic=config.deterministic)

    args.out.mkdir(parents=True, exist_ok=True)
    dataset_audit = audit_dataset(config.dataset)
    write_json(args.out / ARTIFACT_FILENAMES["dataset_audit"], dataset_audit)

    train_dataset, train_loader, val_loader = build_loaders(config, device)
    model = build_model(
        arch=config.arch,
        num_classes=len(train_dataset.classes),
        pretrained=config.pretrained,
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    checkpoint_writer = CheckpointWriter(config.out)
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
        mode="max",
    )

    metadata = make_metadata(
        config=config,
        dataset=train_dataset,
        model=model,
        optimizer=optimizer,
        device=device,
        dataset_audit=dataset_audit,
    )
    write_json(config.out / ARTIFACT_FILENAMES["training_config"], metadata)

    params = metadata["parameter_summary"]
    print(f"device={device} amp={config.amp}")
    print(f"dataset classes={dataset_audit['class_count']} images={dataset_audit['image_count']}")
    print(f"arch={config.arch} pretrained={config.pretrained} freeze_backbone={config.freeze_backbone}")
    print(f"params total={format_large_int(params['total'])} trainable={format_large_int(params['trainable'])}")
    print(f"best_metric={config.metric_for_best}")

    history: list[EpochMetrics] = []
    best_state = None
    best_summary = None
    best_checkpoint_path = None

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            classes=train_dataset.classes,
            optimizer=optimizer,
            scaler=scaler,
            amp=config.amp,
        )
        val_result = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            classes=train_dataset.classes,
            optimizer=None,
            scaler=None,
            amp=config.amp,
        )
        scheduler.step()

        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_result.loss,
            train_top1=train_result.top1,
            train_top3=train_result.top3,
            val_loss=val_result.loss,
            val_top1=val_result.top1,
            val_top3=val_result.top3,
            val_macro_f1=val_result.summary.macro_f1,
            val_weighted_f1=val_result.summary.weighted_f1,
            lr=current_lr(optimizer),
            seconds=time.perf_counter() - start,
        )
        history.append(row)

        monitored_score = choose_best_metric(val_result, config.metric_for_best)
        decision = early_stopping.update(monitored_score, epoch)
        if decision.improved:
            best_state = checkpoint_writer.capture_state(model)
            best_summary = val_result.summary.to_dict()
            best_metadata = metadata | {
                "best_epoch": decision.best_epoch,
                "best_metric_name": config.metric_for_best,
                "best_metric_value": decision.best_score,
                "best_val_acc": val_result.top1,
                "best_val_macro_f1": val_result.macro_f1,
            }
            best_checkpoint_path = checkpoint_writer.save(
                ARTIFACT_FILENAMES["best_checkpoint"],
                model_state=best_state,
                metadata=best_metadata,
                history=history,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_result.loss:.4f} train_top1={train_result.top1:.3f} "
            f"val_loss={val_result.loss:.4f} val_top1={val_result.top1:.3f} "
            f"val_macro_f1={val_result.macro_f1:.3f} "
            f"lr={row.lr:.6f} best={decision.best_score:.3f}@{decision.best_epoch}"
        )

        if decision.should_stop:
            print(f"early_stop patience={config.patience} best_epoch={decision.best_epoch}")
            break

    if not history:
        raise RuntimeError("Training produced no history")

    final_state = checkpoint_writer.capture_state(model)
    final_metadata = metadata | {
        "best_epoch": early_stopping.best_epoch,
        "best_metric_name": config.metric_for_best,
        "best_metric_value": early_stopping.best_score,
    }
    checkpoint_writer.save(
        ARTIFACT_FILENAMES["final_checkpoint"],
        model_state=final_state,
        metadata=final_metadata,
        history=history,
    )

    if best_state is None:
        best_state = final_state
        best_summary = None

    save_history_csv(config.out / ARTIFACT_FILENAMES["history_csv"], history)
    write_json(config.out / ARTIFACT_FILENAMES["history_json"], history)
    write_json(config.out / ARTIFACT_FILENAMES["labels"], train_dataset.class_to_idx)
    write_json(
        config.out / ARTIFACT_FILENAMES["confusion_matrix"],
        {
            "classes": train_dataset.classes,
            "matrix": best_summary["confusion_matrix"] if best_summary else [],
            "summary": best_summary,
        },
    )
    experiment_report = build_experiment_report(
        metadata=final_metadata,
        history=history,
        best_metric_name=config.metric_for_best,
        best_metric_value=early_stopping.best_score,
        best_epoch=early_stopping.best_epoch,
        validation_summary=best_summary,
    )
    write_json(config.out / ARTIFACT_FILENAMES["experiment_report"], experiment_report)
    write_model_card(
        config.out / ARTIFACT_FILENAMES["model_card"],
        metadata=final_metadata,
        best_metric_name=config.metric_for_best,
        best_metric_value=early_stopping.best_score,
        best_epoch=early_stopping.best_epoch,
        validation_summary=best_summary,
    )

    print(f"saved best: {best_checkpoint_path or config.out / ARTIFACT_FILENAMES['best_checkpoint']}")
    print(f"saved final: {config.out / ARTIFACT_FILENAMES['final_checkpoint']}")
    print(f"best_{config.metric_for_best}={early_stopping.best_score:.4f} best_epoch={early_stopping.best_epoch}")


if __name__ == "__main__":
    main()
