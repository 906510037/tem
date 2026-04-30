from __future__ import annotations

import torch

from engine.metrics import accuracy_from_logits, feature_mse, recovery_ratio


def test_accuracy_from_logits() -> None:
    logits = torch.tensor([[2.0, 0.5], [0.2, 1.3], [0.8, 0.1]])
    targets = torch.tensor([0, 1, 1])
    assert abs(accuracy_from_logits(logits, targets) - (2 / 3)) < 1e-6


def test_feature_mse() -> None:
    ref = torch.tensor([1.0, 2.0, 3.0])
    cand = torch.tensor([1.0, 1.0, 5.0])
    assert abs(feature_mse(ref, cand) - (5 / 3)) < 1e-6


def test_recovery_ratio() -> None:
    ratio = recovery_ratio(ideal_acc=0.9, original_acc=0.6, improved_acc=0.75)
    assert abs(ratio - 0.5) < 1e-8
