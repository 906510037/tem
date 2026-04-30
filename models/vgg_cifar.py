"""VGG16 adapted for CIFAR-10 (32x32 input)."""

from __future__ import annotations

from typing import List, Tuple, Union

import torch
from torch import nn

from .factory import register_model
from .injection import InjectionPoints, register_injection


_cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


class VGGCIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        base_channels: int = 64,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.features = self._make_layers(_cfg, batch_norm, base_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _make_layers(
        self, cfg: list, batch_norm: bool, base_channels: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = v
                layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
                )
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@register_model("vgg16_cifar")
def build_vgg16_cifar(num_classes: int = 10, base_channels: int = 64,
                      batch_norm: bool = True) -> VGGCIFAR:
    return VGGCIFAR(num_classes=num_classes, base_channels=base_channels,
                    batch_norm=batch_norm)


@register_injection("vgg16_cifar")
def vgg16_injection(model: VGGCIFAR) -> InjectionPoints:
    """Inject after each MaxPool2d stage — 5 injection points."""
    points: InjectionPoints = []
    for name, module in model.features.named_modules():
        if isinstance(module, nn.MaxPool2d):
            points.append((name, module))
    return points
