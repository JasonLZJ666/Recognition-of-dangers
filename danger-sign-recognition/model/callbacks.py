from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class StopDecision:
    improved: bool
    should_stop: bool
    best_score: float
    best_epoch: int
    stale_epochs: int


class EarlyStopping:
    """Patience based early stopping for validation metrics."""

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max") -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.stale_epochs = 0

    def update(self, score: float, epoch: int) -> StopDecision:
        improved = self._is_improved(score)
        if improved:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1

        return StopDecision(
            improved=improved,
            should_stop=self.stale_epochs >= self.patience,
            best_score=self.best_score,
            best_epoch=self.best_epoch,
            stale_epochs=self.stale_epochs,
        )

    def _is_improved(self, score: float) -> bool:
        if self.mode == "max":
            return float(score) > self.best_score + self.min_delta
        return float(score) < self.best_score - self.min_delta


class CheckpointWriter:
    """Write model checkpoints with consistent metadata and history format."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def capture_state(model: nn.Module) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in model.state_dict().items()}

    def save(
        self,
        filename: str,
        *,
        model_state: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        history: Iterable[Any],
    ) -> Path:
        path = self.out_dir / filename
        torch.save(
            {
                "model": model_state,
                "metadata": metadata,
                "history": [self._to_dict(row) for row in history],
            },
            path,
        )
        return path

    @staticmethod
    def _to_dict(row: Any) -> dict[str, Any]:
        if is_dataclass(row):
            return asdict(row)
        if isinstance(row, dict):
            return row
        raise TypeError(f"Cannot serialize history row of type {type(row).__name__}")
