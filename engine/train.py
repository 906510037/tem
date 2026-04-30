from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LRScheduler
from torch.utils.data import DataLoader

from .metrics import update_accuracy_counts
from .utils import ensure_dir, get_model_output_dir, save_checkpoint


@dataclass
class TrainResult:
    best_accuracy: float
    last_accuracy: float
    best_checkpoint: Path
    last_checkpoint: Path


def _build_scheduler(optimizer: Optimizer, train_cfg: Dict[str, Any]) -> LRScheduler:
    total_epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    main_epochs = total_epochs - warmup_epochs

    if warmup_epochs > 0:
        warmup_start_lr = train_cfg.get("warmup_start_lr", train_cfg["lr"] / 10)
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_lr / train_cfg["lr"],
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        main = CosineAnnealingLR(optimizer, T_max=main_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=total_epochs)


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    train_cfg = config["train"]
    return SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        nesterov=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
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
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
    model_name: str = "model",
) -> TrainResult:
    train_cfg = config["train"]
    out_dirs = get_model_output_dir(config, model_name)
    checkpoint_dir = out_dirs["checkpoint"]
    log_dir = out_dirs["log"]
    last_checkpoint = checkpoint_dir / "last.pt"
    log_path = log_dir / "train.log"
    topk = train_cfg.get("checkpoint_topk", 1)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.get("label_smoothing", 0.0))
    optimizer = build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, train_cfg)
    grad_clip_norm = train_cfg.get("grad_clip_norm", None)

    model.to(device)
    best_accuracy = 0.0
    last_accuracy = 0.0

    # Min-heap of (accuracy, counter, path) — counter breaks ties for heapq
    topk_heap: list[tuple[float, int, Path]] = []
    counter = 0

    best_checkpoint = checkpoint_dir / "best.pt"

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip_norm)
        val_acc = validate(model, test_loader, device)
        scheduler.step()
        last_accuracy = val_acc

        save_checkpoint(last_checkpoint, model, optimizer, epoch, val_acc)

        if val_acc >= best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(best_checkpoint, model, optimizer, epoch, best_accuracy)

        checkpoint_path = checkpoint_dir / f"epoch{epoch:03d}_acc{val_acc:.4f}.pt"
        if len(topk_heap) < topk:
            counter += 1
            heapq.heappush(topk_heap, (val_acc, counter, checkpoint_path))
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_acc)
        elif val_acc > topk_heap[0][0]:
            _, _, old_path = heapq.heappop(topk_heap)
            if old_path.exists():
                old_path.unlink()
            counter += 1
            heapq.heappush(topk_heap, (val_acc, counter, checkpoint_path))
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_acc)

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
