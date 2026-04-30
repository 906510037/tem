from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evaluate import run_noise_scan
from engine.utils import build_cifar10_loaders, load_yaml, load_checkpoint, seed_everything, select_device
from models import resnet18_cifar
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sigma0 noise scan.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--original-config", default="configs/temp_original.yaml")
    parser.add_argument("--improved-config", default="configs/temp_improved_4bin.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/csv/accuracy_vs_noise.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.base_config)
    original_config = load_yaml(args.original_config)
    improved_config = load_yaml(args.improved_config)

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

    frame = run_noise_scan(
        model=model,
        loader=test_loader,
        device=device,
        original_hooks=original_hooks,
        improved_hooks=improved_hooks,
        sigma_values=base_config["eval"]["noise_scan_values"],
        temperature=base_config["eval"]["noise_temperature"],
        noise_seed=base_config["system"]["seed"],
        output_path=args.output,
    )

    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
