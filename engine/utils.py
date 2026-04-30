from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(config_value: str) -> torch.device:
    if config_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_value)


def build_cifar10_loaders(config: Dict[str, Any], train: bool = True, download: bool = True) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    root = dataset_cfg["root"]
    train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)

    generator = torch.Generator()
    generator.manual_seed(config["system"]["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=eval_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
