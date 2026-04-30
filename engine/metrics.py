from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import Tensor


def accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def update_accuracy_counts(logits: Tensor, targets: Tensor) -> tuple[int, int]:
    predictions = logits.argmax(dim=1)
    correct = int((predictions == targets).sum().item())
    total = int(targets.numel())
    return correct, total


def feature_mse(reference: Tensor, candidate: Tensor) -> float:
    return torch.mean((reference - candidate) ** 2).item()


def recovery_ratio(ideal_acc: float, original_acc: float, improved_acc: float) -> float:
    orig_drop = ideal_acc - original_acc
    imp_drop = ideal_acc - improved_acc
    if abs(orig_drop) < 1e-12:
        return 0.0
    return (orig_drop - imp_drop) / orig_drop


def aggregate_feature_mse(squared_error_sums: Dict[str, float], counts: Dict[str, int]) -> Dict[str, float]:
    return {
        name: squared_error_sums[name] / max(counts[name], 1)
        for name in squared_error_sums
    }
