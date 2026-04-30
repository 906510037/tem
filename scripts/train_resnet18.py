from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.train import fit
from engine.utils import build_cifar10_loaders, ensure_dir, load_yaml, seed_everything, select_device
from models import resnet18_cifar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 ResNet18.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to base config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(config["system"]["seed"], config["system"]["deterministic"])
    device = select_device(config["system"]["device"])

    for key in ("checkpoint_dir", "csv_dir", "figure_dir", "log_dir"):
        ensure_dir(config["outputs"][key])

    train_loader, test_loader = build_cifar10_loaders(config, train=True)
    model = resnet18_cifar(
        num_classes=config["dataset"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    )
    result = fit(model, train_loader, test_loader, config, device)
    print(f"best_checkpoint={result.best_checkpoint}")
    print(f"last_checkpoint={result.last_checkpoint}")
    print(f"best_accuracy={result.best_accuracy:.4f}")
    print(f"last_accuracy={result.last_accuracy:.4f}")


if __name__ == "__main__":
    main()
