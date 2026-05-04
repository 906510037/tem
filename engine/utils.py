from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
#  YAML / config helpers
# ---------------------------------------------------------------------------

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


def load_experiment_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load and validate the unified experiment config."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
    config = load_yaml(path)
    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    required = ["system", "dataset", "models", "train", "eval",
                 "temperature", "nonideal", "outputs"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    tcfg = config["temperature"]
    t_min, t_max = tcfg["range"]
    t_step = tcfg["step"]
    n_points = (t_max - t_min) / t_step
    if not n_points.is_integer():
        raise ValueError(
            f"Temperature step {t_step} does not divide "
            f"range [{t_min}, {t_max}] evenly"
        )
    if t_max > 110:
        raise ValueError(f"Maximum temperature {t_max} exceeds cap of 110 C")

    for model_name in config["models"]["active"]:
        if model_name not in config["models"]:
            raise ValueError(f"Active model '{model_name}' has no config entry")


def get_temperature_points(config: Dict[str, Any]) -> list[float]:
    tcfg = config["temperature"]
    t_min, t_max = tcfg["range"]
    t_step = tcfg["step"]
    vals = np.arange(t_min, t_max + t_step / 2, t_step)
    return [round(float(v), 2) for v in vals]


def get_model_output_dir(config: Dict[str, Any], model_name: str) -> Dict[str, Path]:
    root = Path(config["outputs"]["root"]) / model_name
    dirs = {
        "checkpoint": ensure_dir(root / "checkpoints"),
        "csv": ensure_dir(root / "csv"),
        "figure": ensure_dir(root / "figures"),
        "log": ensure_dir(root / "log"),
    }
    return dirs


# ---------------------------------------------------------------------------
#  Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
#  Reproducibility / device
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Augmentation
# ---------------------------------------------------------------------------

class Cutout:
    """Randomly mask out one square region of the input tensor."""
    def __init__(self, length: int):
        self.length = length

    def __call__(self, img: Tensor) -> Tensor:
        _, h, w = img.shape
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        half = self.length // 2
        y1 = max(0, y - half)
        y2 = min(h, y + half)
        x1 = max(0, x - half)
        x2 = min(w, x + half)
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self) -> str:
        return f"Cutout(length={self.length})"


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def build_cifar10_loaders(
    config: Dict[str, Any],
    train: bool = True,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)

    base_train_xf: list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    base_test_xf: list = []

    train_xf = base_train_xf + [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if train_cfg.get("cutout", False):
        train_xf.append(Cutout(train_cfg.get("cutout_length", 16)))

    test_xf = base_test_xf + [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    root = dataset_cfg["root"]
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=download, transform=transforms.Compose(train_xf),
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=download, transform=transforms.Compose(test_xf),
    )

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


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_acc: float,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_acc": best_acc,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
