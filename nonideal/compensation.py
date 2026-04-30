"""温度分档补偿逻辑。

这里不直接处理张量，而是负责回答两个问题：
1. 当前温度落在哪个补偿区间
2. 这个区间对应的中心温度是多少
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CompensationSpec:
    """描述补偿档位边界，并提供温度到档位的映射能力。"""

    bin_edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.bin_edges) < 2:
            raise ValueError("bin_edges must contain at least two values.")
        if any(right <= left for left, right in zip(self.bin_edges, self.bin_edges[1:])):
            raise ValueError("bin_edges must be strictly increasing.")

    @classmethod
    def from_sequence(cls, edges: Sequence[float]) -> "CompensationSpec":
        return cls(tuple(float(edge) for edge in edges))

    def num_bins(self) -> int:
        return len(self.bin_edges) - 1

    def get_bin_index(self, temperature: float) -> int:
        """返回温度所属的档位编号。"""
        if temperature <= self.bin_edges[0]:
            return 0
        if temperature >= self.bin_edges[-1]:
            return self.num_bins() - 1
        for index, (left, right) in enumerate(zip(self.bin_edges, self.bin_edges[1:])):
            if left <= temperature < right:
                return index
        return self.num_bins() - 1

    def get_bin_center(self, temperature: float) -> float:
        """返回当前温度所在档位的中心温度 Tc。"""
        index = self.get_bin_index(temperature)
        left = self.bin_edges[index]
        right = self.bin_edges[index + 1]
        return 0.5 * (left + right)
