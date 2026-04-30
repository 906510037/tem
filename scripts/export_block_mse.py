from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from engine.evaluate import evaluate_blockwise_mse
from engine.utils import build_cifar10_loaders, ensure_dir, load_yaml, load_checkpoint, seed_everything, select_device
from models import resnet18_cifar
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export block-wise MSE.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--original-config", default="configs/temp_original.yaml")
    parser.add_argument("--improved-config", default="configs/temp_improved_4bin.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--output", default="outputs/csv/blockwise_mse.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.base_config)
    original_config = load_yaml(args.original_config)
    improved_config = load_yaml(args.improved_config)
    temperature = args.temperature or base_config["eval"]["block_mse_temperature"]

    seed_everything(base_config["system"]["seed"], base_config["system"]["deterministic"])
    device = select_device(base_config["system"]["device"])
    _, test_loader = build_cifar10_loaders(base_config, train=False)

    model = resnet18_cifar(
        num_classes=base_config["dataset"]["num_classes"],
        base_channels=base_config["model"]["base_channels"],
    )
    load_checkpoint(args.checkpoint, model)
    model.to(device)

    blocks = list(model.iter_residual_blocks())
    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    original_hooks = NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(NonIdealConfig.from_dict(original_config), noise_seed=base_config["system"]["seed"]),
        mode="replace_with_nonideal",
    )
    improved_hooks = NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(NonIdealConfig.from_dict(improved_config), noise_seed=base_config["system"]["seed"]),
        mode="replace_with_nonideal",
    )
    original_mse = evaluate_blockwise_mse(
        model,
        test_loader,
        device,
        ideal_hooks,
        original_hooks,
        temperature,
        base_config["system"]["seed"],
    )
    improved_mse = evaluate_blockwise_mse(
        model,
        test_loader,
        device,
        ideal_hooks,
        improved_hooks,
        temperature,
        base_config["system"]["seed"],
    )

    rows = []
    for block_name in OrderedDict.fromkeys(list(original_mse.keys()) + list(improved_mse.keys())):
        rows.append(
            {
                "block": block_name,
                "temperature": temperature,
                "original_mse": original_mse[block_name],
                "improved_mse": improved_mse[block_name],
            }
        )

    frame = pd.DataFrame(rows)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
