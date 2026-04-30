"""Bin sweep using the unified experiment config.

Usage:
    python scripts/eval_bin_sweep.py
    python scripts/eval_bin_sweep.py --model resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evaluate import run_bin_sweep
from engine.utils import (
    build_cifar10_loaders,
    get_model_output_dir,
    get_temperature_points,
    load_checkpoint,
    load_experiment_config,
    seed_everything,
    select_device,
)
from models import create_model, get_injection_points
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def main():
    parser = argparse.ArgumentParser(description="Run compensation bin sweep.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or (ROOT / "configs" / "experiment.yaml")
    config = load_experiment_config(Path(config_path))
    model_name = args.model or config["models"]["active"][0]
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    system_cfg = config["system"]
    seed = system_cfg["seed"]
    seed_everything(seed, system_cfg["deterministic"])
    device = select_device(system_cfg["device"])

    out_dirs = get_model_output_dir(config, model_name)
    checkpoint_path = args.checkpoint or str(out_dirs["checkpoint"] / "best.pt")

    _, test_loader = build_cifar10_loaders(config, train=False)
    model = create_model(model_name, model_cfg,
                         num_classes=config["dataset"]["num_classes"])
    load_checkpoint(checkpoint_path, model)
    model.to(device)

    bin_configs = config["nonideal"]["bins"]
    active_sweep = bin_configs.get("active_sweep", ["2", "4", "6"])

    blocks = get_injection_points(model, arch)
    hook_managers = {}
    for label in active_sweep:
        imp = dict(config["nonideal"]["improved"])
        imp["bin_edges"] = bin_configs[label]
        hook_managers[label] = NonIdealHookManager(
            blocks,
            TemperatureNonIdeality(NonIdealConfig.from_dict(imp), noise_seed=seed),
            mode="replace_with_nonideal",
        )

    tcfg = config["temperature"]
    temperatures = (
        get_temperature_points(config)
        if tcfg.get("bin_sweep_all_temps", True)
        else tcfg["block_mse_temperature"]
    )
    frame = run_bin_sweep(
        model, test_loader, device, hook_managers,
        temperatures, seed,
        output_path=args.output or str(out_dirs["csv"] / "bin_sweep.csv"),
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
