from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset: Path
    out: Path
    arch: str
    pretrained: bool
    freeze_backbone: bool
    epochs: int
    batch_size: int
    image_size: int
    repeats: int
    lr: float
    weight_decay: float
    dropout: float
    label_smoothing: float
    patience: int
    min_delta: float
    metric_for_best: str
    seed: int
    workers: int
    auto_augment: bool
    amp: bool
    deterministic: bool

    @classmethod
    def from_args(cls, args, amp: bool) -> "TrainingConfig":
        return cls(
            dataset=args.dataset,
            out=args.out,
            arch=args.arch,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            repeats=args.repeats,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            patience=args.patience,
            min_delta=args.min_delta,
            metric_for_best=args.metric_for_best,
            seed=args.seed,
            workers=args.workers,
            auto_augment=not args.no_auto_augment,
            amp=amp,
            deterministic=args.deterministic,
        )

    def to_json_dict(self) -> dict:
        data = asdict(self)
        data["dataset"] = str(self.dataset)
        data["out"] = str(self.out)
        return data
