"""Injection point registry for different model architectures.

Each model architecture registers a function that returns a list of
(name, module) tuples.  The hook manager attaches to these modules.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from torch import nn

InjectionPoints = List[Tuple[str, nn.Module]]
InjectionFn = Callable[[nn.Module], InjectionPoints]

INJECTION_REGISTRY: Dict[str, InjectionFn] = {}


def register_injection(name: str):
    """Decorator to register an injection-point extractor for an architecture."""
    def decorator(fn: InjectionFn) -> InjectionFn:
        INJECTION_REGISTRY[name] = fn
        return fn
    return decorator


def get_injection_points(model: nn.Module, arch: str) -> InjectionPoints:
    if arch not in INJECTION_REGISTRY:
        raise KeyError(
            f"No injection function registered for '{arch}'. "
            f"Available: {list(INJECTION_REGISTRY.keys())}"
        )
    return INJECTION_REGISTRY[arch](model)
