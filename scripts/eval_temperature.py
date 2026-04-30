"""Temperature accuracy scan using the unified experiment config.

Usage:
    python scripts/eval_temperature.py
    python scripts/eval_temperature.py --model resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evaluate import run_temperature_scan
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
    parser = argparse.ArgumentParser(description="Run temperature accuracy scan.")
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

    blocks = get_injection_points(model, arch)
    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    orig_hooks = NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(
            NonIdealConfig.from_dict(config["nonideal"]["original"]), noise_seed=seed),
        mode="replace_with_nonideal",
    )
    imp_hooks = NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(
            NonIdealConfig.from_dict(config["nonideal"]["improved"]), noise_seed=seed),
        mode="replace_with_nonideal",
    )

    frame = run_temperature_scan(
        model, test_loader, device,
        ideal_hooks, orig_hooks, imp_hooks,
        get_temperature_points(config), seed,
        output_path=args.output or str(out_dirs["csv"] / "accuracy_vs_temp.csv"),
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
