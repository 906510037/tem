"""Residual block 输出 hook 管理器。

这个模块负责两件事：
1. 抓取每个 residual block 的输出特征
2. 在 block 最终输出后替换为带非理想行为的新特征
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .temp_model import TemperatureNonIdeality

HookMode = Literal["capture_only", "replace_with_nonideal"]


@dataclass
class HookState:
    """保存 hook 在一次评估中的工作状态。"""

    temperature: float = 25.0
    enabled: bool = True


class NonIdealHookManager:
    """统一管理一组 residual block 的 forward hook。"""

    def __init__(
        self,
        blocks: Iterable[Tuple[str, nn.Module]],
        nonideality: Optional[TemperatureNonIdeality] = None,
        mode: HookMode = "capture_only",
    ) -> None:
        self.blocks = list(blocks)
        self.nonideality = nonideality
        self.mode = mode
        self.state = HookState()
        self._handles: List[RemovableHandle] = []
        self._captured: "OrderedDict[str, Tensor]" = OrderedDict()

    def attach(self) -> None:
        """把 hook 注册到所有 residual block 上。"""
        if self._handles:
            return
        for name, module in self.blocks:
            handle = module.register_forward_hook(self._build_hook(name))
            self._handles.append(handle)

    def remove(self) -> None:
        """移除所有 hook，避免不同评估模式之间互相干扰。"""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self._captured.clear()

    def set_temperature(self, temperature: float) -> None:
        self.state.temperature = float(temperature)

    def set_mode(self, mode: HookMode) -> None:
        self.mode = mode

    def set_enabled(self, enabled: bool) -> None:
        self.state.enabled = enabled

    def set_noise_seed(self, seed: int) -> None:
        if self.nonideality is not None:
            self.nonideality.reset_noise_seed(seed)

    def last_features(self) -> Dict[str, Tensor]:
        """返回最近一次 forward 过程中捕获到的 block 输出。"""
        return OrderedDict((name, tensor) for name, tensor in self._captured.items())

    def _build_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: Tuple[Tensor, ...], output: Tensor) -> Tensor | None:
            if not self.state.enabled:
                self._captured[name] = output.detach()
                return None
            if self.mode == "replace_with_nonideal":
                if self.nonideality is None:
                    raise RuntimeError("replace_with_nonideal mode requires a nonideality module.")
                # 注入位置就是 residual block 的最终输出之后。
                output = self.nonideality(output, self.state.temperature)
            self._captured[name] = output.detach()
            if self.mode == "replace_with_nonideal":
                return output
            return None

        return hook


@contextmanager
def hook_session(
    hook_manager: NonIdealHookManager,
    temperature: float = 25.0,
    noise_seed: int | None = None,
) -> Generator[NonIdealHookManager, None, None]:
    """Context manager that attaches hooks and guarantees cleanup."""
    hook_manager.attach()
    hook_manager.set_temperature(temperature)
    hook_manager.clear()
    if noise_seed is not None:
        hook_manager.set_noise_seed(noise_seed)
    try:
        yield hook_manager
    finally:
        hook_manager.remove()
