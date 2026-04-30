from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import aggregate_feature_mse, recovery_ratio, update_accuracy_counts
from .utils import ensure_dir
from nonideal.hooks import NonIdealHookManager


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hook_manager: NonIdealHookManager | None = None,
    temperature: float = 25.0,
    noise_seed: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0

    if hook_manager is not None:
        hook_manager.attach()
        hook_manager.set_temperature(temperature)
        hook_manager.clear()
        if noise_seed is not None:
            hook_manager.set_noise_seed(noise_seed)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        batch_correct, batch_total = update_accuracy_counts(logits, targets)
        correct += batch_correct
        total += batch_total

    if hook_manager is not None:
        hook_manager.remove()

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_blockwise_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ideal_hooks: NonIdealHookManager,
    nonideal_hooks: NonIdealHookManager,
    temperature: float,
    noise_seed: int,
) -> Dict[str, float]:
    model.eval()
    squared_error_sums: Dict[str, float] = OrderedDict()
    counts: Dict[str, int] = OrderedDict()

    for images, _targets in loader:
        images = images.to(device, non_blocking=True)

        ideal_hooks.attach()
        ideal_hooks.set_mode("capture_only")
        ideal_hooks.set_temperature(temperature)
        ideal_hooks.clear()
        _ = model(images)
        ideal_features = ideal_hooks.last_features()
        ideal_hooks.remove()

        nonideal_hooks.attach()
        nonideal_hooks.set_mode("replace_with_nonideal")
        nonideal_hooks.set_temperature(temperature)
        nonideal_hooks.set_noise_seed(noise_seed)
        nonideal_hooks.clear()
        _ = model(images)
        nonideal_features = nonideal_hooks.last_features()
        nonideal_hooks.remove()

        for name, ideal_feature in ideal_features.items():
            candidate = nonideal_features[name]
            squared_error = torch.sum((ideal_feature - candidate) ** 2).item()
            numel = int(ideal_feature.numel())
            squared_error_sums[name] = squared_error_sums.get(name, 0.0) + squared_error
            counts[name] = counts.get(name, 0) + numel

    return aggregate_feature_mse(squared_error_sums, counts)


def run_temperature_scan(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ideal_hooks: NonIdealHookManager | None,
    original_hooks: NonIdealHookManager,
    improved_hooks: NonIdealHookManager,
    temperatures: Sequence[float],
    noise_seed: int,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    rows = []
    for temperature in temperatures:
        ideal_acc = evaluate_accuracy(model, loader, device, ideal_hooks, temperature, noise_seed)
        original_acc = evaluate_accuracy(model, loader, device, original_hooks, temperature, noise_seed)
        improved_acc = evaluate_accuracy(model, loader, device, improved_hooks, temperature, noise_seed)
        rows.append(
            {
                "temperature": temperature,
                "ideal": ideal_acc,
                "original": original_acc,
                "improved": improved_acc,
                "recovery_ratio": recovery_ratio(ideal_acc, original_acc, improved_acc),
            }
        )
    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        frame.to_csv(output_path, index=False)
    return frame


def run_noise_scan(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    original_hooks: NonIdealHookManager,
    improved_hooks: NonIdealHookManager,
    sigma_values: Iterable[float],
    temperature: float,
    noise_seed: int,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    original_module = original_hooks.nonideality
    improved_module = improved_hooks.nonideality
    if original_module is None or improved_module is None:
        raise ValueError("Noise scan requires original and improved nonideality modules.")

    rows = []
    original_base_sigma = original_module.config.sigma0
    improved_base_sigma = improved_module.config.sigma0

    try:
        for sigma0 in sigma_values:
            original_module.config = type(original_module.config)(**{**original_module.config.__dict__, "sigma0": sigma0})
            improved_module.config = type(improved_module.config)(**{**improved_module.config.__dict__, "sigma0": sigma0})
            original_acc = evaluate_accuracy(model, loader, device, original_hooks, temperature, noise_seed)
            improved_acc = evaluate_accuracy(model, loader, device, improved_hooks, temperature, noise_seed)
            rows.append(
                {
                    "sigma": sigma0,
                    "original": original_acc,
                    "improved": improved_acc,
                }
            )
    finally:
        original_module.config = type(original_module.config)(**{**original_module.config.__dict__, "sigma0": original_base_sigma})
        improved_module.config = type(improved_module.config)(**{**improved_module.config.__dict__, "sigma0": improved_base_sigma})

    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        frame.to_csv(output_path, index=False)
    return frame


def run_bin_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hook_managers: Dict[str, NonIdealHookManager],
    temperature: float | Sequence[float],
    noise_seed: int,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    rows = []
    if isinstance(temperature, (int, float)):
        temperatures = [float(temperature)]
    else:
        temperatures = [float(item) for item in temperature]

    for label, hooks in hook_managers.items():
        accuracies = []
        for temp in temperatures:
            accuracy = evaluate_accuracy(model, loader, device, hooks, temp, noise_seed)
            accuracies.append(accuracy)
            rows.append({"bins": label, "accuracy": accuracy, "temperature": temp})

        if len(temperatures) > 1:
            rows.append(
                {
                    "bins": label,
                    "accuracy": sum(accuracies) / len(accuracies),
                    "temperature": "mean",
                }
            )

    frame = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        frame.to_csv(output_path, index=False)
    return frame
