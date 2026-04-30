from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .metrics import update_accuracy_counts
from .utils import ensure_dir, save_checkpoint


@dataclass
class TrainResult:
    best_accuracy: float
    last_accuracy: float
    best_checkpoint: Path
    last_checkpoint: Path


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    train_cfg = config["train"]
    return SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_correct, batch_total = update_accuracy_counts(logits.detach(), targets)
        correct += batch_correct
        total += batch_total

    return correct / max(total, 1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        batch_correct, batch_total = update_accuracy_counts(logits, targets)
        correct += batch_correct
        total += batch_total
    return correct / max(total, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
) -> TrainResult:
    outputs = config["outputs"]
    checkpoint_dir = ensure_dir(outputs["checkpoint_dir"])
    log_dir = ensure_dir(outputs["log_dir"])
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    log_path = log_dir / "train.log"

    train_cfg = config["train"]
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    optimizer = build_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    model.to(device)
    best_accuracy = 0.0
    last_accuracy = 0.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, test_loader, device)
        scheduler.step()
        last_accuracy = val_acc

        save_checkpoint(last_checkpoint, model, optimizer, epoch, best_accuracy)
        if val_acc >= best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(best_checkpoint, model, optimizer, epoch, best_accuracy)

        message = (
            f"epoch={epoch:03d} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        print(message)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return TrainResult(
        best_accuracy=best_accuracy,
        last_accuracy=last_accuracy,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )
