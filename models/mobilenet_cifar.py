"""MobileNetV2 adapted for CIFAR-10 (32x32 input)."""

from __future__ import annotations

import torch
from torch import nn

from .factory import register_model
from .injection import InjectionPoints, register_injection


def _make_divisible(v: float, divisor: int = 8) -> int:
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: float) -> None:
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_residual = (stride == 1 and inp == oup)

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                       groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0) -> None:
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult))

        features: list[nn.Module] = [ConvBNReLU(3, input_channel, stride=1)]
        for t, c, n, s in inverted_residual_settings:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._init_weights()

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


@register_model("mobilenetv2_cifar")
def build_mobilenetv2_cifar(num_classes: int = 10, width_mult: float = 1.0
                            ) -> MobileNetV2CIFAR:
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult)


@register_injection("mobilenetv2_cifar")
def mobilenetv2_injection(model: MobileNetV2CIFAR) -> InjectionPoints:
    """Inject after each InvertedResidual block."""
    points: InjectionPoints = []
    for name, module in model.features.named_modules():
        if isinstance(module, InvertedResidual):
            points.append((name, module))
    return points
