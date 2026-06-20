from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from model.model_utils import freeze_modules


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention used by the custom CNN."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(self.pool(x))


class DepthwiseSeparableBlock(nn.Module):
    """Mobile-style block with depthwise conv, attention and residual path."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvNormAct(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels),
            SqueezeExcitation(in_channels),
            ConvNormAct(in_channels, out_channels, kernel_size=1, activation=False),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return self.activation(out)


class StrongCNN(nn.Module):
    """Custom CNN for small offline datasets.

    The network combines a compact stem, depthwise-separable residual blocks,
    squeeze-excitation channel attention, global pooling and a two-layer head.
    It is slower than a toy CNN but gives the project a meaningful custom
    architecture when pretrained weights are unavailable.
    """

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        widths = (48, 96, 160, 224)
        layers: list[nn.Module] = [
            ConvNormAct(3, 32, kernel_size=3, stride=2),
            ConvNormAct(32, 32, kernel_size=3, stride=1),
        ]
        in_channels = 32
        for width in widths:
            layers.extend(
                [
                    DepthwiseSeparableBlock(in_channels, width, stride=2, dropout=dropout * 0.30),
                    DepthwiseSeparableBlock(width, width, stride=1, dropout=dropout * 0.15),
                ]
            )
            in_channels = width

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(widths[-1]),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)


def _replace_efficientnet_head(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


def _replace_resnet_head(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


def _replace_mobilenet_head(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


def build_model(arch: str, num_classes: int, pretrained: bool, dropout: float, freeze_backbone: bool) -> nn.Module:
    arch = arch.lower()
    if arch == "strong_cnn":
        return StrongCNN(num_classes=num_classes, dropout=dropout)

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            freeze_modules([model.features])
        return _replace_efficientnet_head(model, num_classes, dropout)

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("layer4") or name.startswith("fc")
        return _replace_resnet_head(model, num_classes, dropout)

    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        if freeze_backbone:
            freeze_modules([model.features])
        return _replace_mobilenet_head(model, num_classes, dropout)

    raise ValueError(f"Unsupported arch: {arch}")
