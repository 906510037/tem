"""Model factory for creating models by registered name."""

from __future__ import annotations

from typing import Any, Callable, Dict

from torch import nn

_MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model builder function."""
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _MODEL_BUILDERS[name] = fn
        return fn
    return decorator


def create_model(name: str, model_cfg: Dict[str, Any],
                 num_classes: int = 10) -> nn.Module:
    if name not in _MODEL_BUILDERS:
        raise KeyError(
            f"Unknown model: '{name}'. "
            f"Available: {list(_MODEL_BUILDERS.keys())}"
        )
    params = {k: v for k, v in model_cfg.items() if k != "arch"}
    return _MODEL_BUILDERS[name](num_classes=num_classes, **params)


def list_models() -> list[str]:
    return sorted(_MODEL_BUILDERS.keys())
