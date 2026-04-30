from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evaluate import run_bin_sweep
from engine.utils import build_cifar10_loaders, load_yaml, load_checkpoint, seed_everything, select_device
from models import resnet18_cifar
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compensation bin sweep over the configured temperatures.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config-2bin", default="configs/temp_improved_2bin.yaml")
    parser.add_argument("--config-4bin", default="configs/temp_improved_4bin.yaml")
    parser.add_argument("--config-6bin", default="configs/temp_improved_6bin.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/csv/bin_sweep.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.base_config)
    config_2bin = load_yaml(args.config_2bin)
    config_4bin = load_yaml(args.config_4bin)
    config_6bin = load_yaml(args.config_6bin)

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
    hook_managers = {
        "2": NonIdealHookManager(
            blocks,
            TemperatureNonIdeality(NonIdealConfig.from_dict(config_2bin), noise_seed=base_config["system"]["seed"]),
            mode="replace_with_nonideal",
        ),
        "4": NonIdealHookManager(
            blocks,
            TemperatureNonIdeality(NonIdealConfig.from_dict(config_4bin), noise_seed=base_config["system"]["seed"]),
            mode="replace_with_nonideal",
        ),
        "6": NonIdealHookManager(
            blocks,
            TemperatureNonIdeality(NonIdealConfig.from_dict(config_6bin), noise_seed=base_config["system"]["seed"]),
            mode="replace_with_nonideal",
        ),
    }

    eval_cfg = base_config["eval"]
    temperatures = (
        eval_cfg["temperature_points"]
        if eval_cfg.get("bin_sweep_all_temps", True)
        else eval_cfg["bin_sweep_temperature"]
    )
    frame = run_bin_sweep(
        model=model,
        loader=test_loader,
        device=device,
        hook_managers=hook_managers,
        temperature=temperatures,
        noise_seed=base_config["system"]["seed"],
        output_path=args.output,
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
