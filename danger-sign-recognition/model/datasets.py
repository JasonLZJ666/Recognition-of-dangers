from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.constants import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class AugmentedSignDataset(Dataset):
    """Folder dataset with repeat-based virtual expansion.

    The starter dataset has one clean source image per class. Repeating the
    source samples while applying random transforms lets the network see many
    variants. Add real scene images under dataset/<class_name>/ to improve
    deployment accuracy.
    """

    def __init__(
        self,
        root: Path,
        image_size: int,
        repeats: int,
        training: bool,
        auto_augment: bool,
    ) -> None:
        self.root = root
        self.classes = sorted(path.name for path in root.iterdir() if path.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[Sample] = []
        for class_name in self.classes:
            for path in sorted((root / class_name).glob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append(Sample(path, self.class_to_idx[class_name]))
        if len(self.classes) < 2:
            raise RuntimeError(f"Need at least two class folders under {root}")
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

        self.training = training
        self.repeats = repeats if training else max(1, min(12, repeats // 6))
        self.transform = (
            self._training_transform(image_size, auto_augment)
            if training
            else self._validation_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index % len(self.samples)]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label

    @staticmethod
    def _training_transform(image_size: int, auto_augment: bool) -> transforms.Compose:
        steps: list[Callable] = [
            transforms.Resize((image_size + 28, image_size + 28)),
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.88, 1.12)),
            transforms.RandomApply([transforms.ColorJitter(0.28, 0.28, 0.20, 0.05)], p=0.85),
            transforms.RandomAffine(
                degrees=16,
                translate=(0.10, 0.10),
                scale=(0.82, 1.18),
                shear=5,
                fill=255,
            ),
            transforms.RandomPerspective(distortion_scale=0.18, p=0.35, fill=255),
        ]
        if auto_augment and hasattr(transforms, "RandAugment"):
            steps.append(transforms.RandAugment(num_ops=2, magnitude=8))
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.18, scale=(0.02, 0.08), ratio=(0.4, 2.2), value=1.0),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        return transforms.Compose(steps)

    @staticmethod
    def _validation_transform(image_size: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

