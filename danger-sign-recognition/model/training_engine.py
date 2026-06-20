from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.metrics import AverageMeter, EvaluationSummary, summarize_classification, topk_correct


@dataclass
class RunEpochResult:
    loss: float
    top1: float
    top3: float
    macro_f1: float
    y_true: list[int]
    y_pred: list[int]
    summary: EvaluationSummary


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_top1: float
    train_top3: float
    val_loss: float
    val_top1: float
    val_top3: float
    val_macro_f1: float
    val_weighted_f1: float
    lr: float
    seconds: float


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: Sequence[str],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
) -> RunEpochResult:
    training = optimizer is not None
    model.train(training)
    loss_meter = AverageMeter("loss")
    top1_meter = AverageMeter("top1")
    top3_meter = AverageMeter("top3")
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = labels.size(0)

            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            predictions = logits.argmax(dim=1)
            topk = topk_correct(logits.detach(), labels.detach(), topk=(1, 3))
            loss_meter.update(float(loss.item()), batch_size)
            top1_meter.update(topk[1] / batch_size, batch_size)
            top3_meter.update(topk[3] / batch_size, batch_size)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    summary = summarize_classification(
        loss=loss_meter.avg,
        top1=top1_meter.avg,
        top3=top3_meter.avg,
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
    )
    return RunEpochResult(
        loss=summary.loss,
        top1=summary.top1,
        top3=summary.top3,
        macro_f1=summary.macro_f1,
        y_true=y_true,
        y_pred=y_pred,
        summary=summary,
    )


def choose_best_metric(result: RunEpochResult, metric_name: str) -> float:
    if metric_name == "val_top1":
        return result.top1
    if metric_name == "val_macro_f1":
        return result.macro_f1
    if metric_name == "val_top3":
        return result.top3
    raise ValueError(f"Unsupported metric for best checkpoint: {metric_name}")
