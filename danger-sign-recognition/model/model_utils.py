from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    frozen: int
    trainable_ratio: float

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "trainable": self.trainable,
            "frozen": self.frozen,
            "trainable_ratio": self.trainable_ratio,
        }


def count_parameters(model: nn.Module) -> ParameterSummary:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    ratio = trainable / total if total else 0.0
    return ParameterSummary(total=total, trainable=trainable, frozen=frozen, trainable_ratio=ratio)


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)


def snapshot_environment(device: torch.device, amp: bool) -> dict:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_name,
        "device": str(device),
        "amp": bool(amp),
    }


def describe_optimizer(optimizer: torch.optim.Optimizer) -> dict:
    group_rows = []
    for index, group in enumerate(optimizer.param_groups):
        group_rows.append(
            {
                "group": index,
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "params": sum(param.numel() for param in group.get("params", []) if param.requires_grad),
            }
        )
    return {"name": optimizer.__class__.__name__, "param_groups": group_rows}


def format_large_int(value: int) -> str:
    return f"{value:,}"


def collect_trainable_layer_names(model: nn.Module, limit: int = 24) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def freeze_modules(modules: Iterable[nn.Module]) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = False
