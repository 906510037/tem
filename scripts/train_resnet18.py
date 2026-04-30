"""Train a model on CIFAR-10 using the unified experiment config.

Usage:
    python scripts/train_resnet18.py
    python scripts/train_resnet18.py --model resnet34
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.train import fit
from engine.utils import (
    build_cifar10_loaders,
    load_experiment_config,
    seed_everything,
    select_device,
)
from models import create_model


def main():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR-10.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or (ROOT / "configs" / "experiment.yaml")
    config = load_experiment_config(Path(config_path))
    model_name = args.model or config["models"]["active"][0]

    system_cfg = config["system"]
    seed_everything(system_cfg["seed"], system_cfg["deterministic"])
    device = select_device(system_cfg["device"])

    train_loader, test_loader = build_cifar10_loaders(config, train=True)
    model = create_model(model_name, config["models"][model_name],
                         num_classes=config["dataset"]["num_classes"])
    result = fit(model, train_loader, test_loader, config, device,
                 model_name=model_name)
    print(f"best_checkpoint: {result.best_checkpoint}")
    print(f"best_accuracy:   {result.best_accuracy:.4f}")


if __name__ == "__main__":
    main()
