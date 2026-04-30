"""CIFAR-10 版本的 ResNet18。

这个文件的目标是提供一个适合 32x32 输入的残差网络，并且把每个
residual block 暴露出来，方便后续在 block 最终输出后注册 hook。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor, nn

from .blocks import BasicBlock


class CIFARResNet(nn.Module):
    """适配 CIFAR-10 输入尺寸的 ResNet。

    与 ImageNet 版本不同，这里不使用 7x7 大卷积和 maxpool 开头，
    而是改用更适合 32x32 图像的 3x3 stem。
    """

    def __init__(
        self,
        block: type[BasicBlock] = BasicBlock,
        layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_classes: int = 10,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: List[nn.Module] = [
            block(self.in_channels, out_channels, stride=stride),
        ]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def iter_residual_blocks(self) -> Iterable[Tuple[str, nn.Module]]:
        """按固定顺序暴露所有 residual block，便于统一注入非理想行为。"""
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self, layer_name)
            for index, block in enumerate(layer):
                yield f"{layer_name}.{index}", block

    def get_residual_block_names(self) -> List[str]:
        return [name for name, _ in self.iter_residual_blocks()]

    def forward_features(self, x: Tensor) -> Tuple[Tensor, OrderedDict[str, Tensor]]:
        """返回最终特征和每个 residual block 的输出。"""
        features: OrderedDict[str, Tensor] = OrderedDict()
        x = self.stem(x)
        for name, block in self.iter_residual_blocks():
            x = block(x)
            features[name] = x
        return x, features

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        x, features = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        if return_features:
            return logits, features
        return logits


def resnet18_cifar(num_classes: int = 10, base_channels: int = 64) -> CIFARResNet:
    """构造一个标准的 CIFAR-10 ResNet18。"""
    return CIFARResNet(num_classes=num_classes, base_channels=base_channels)
