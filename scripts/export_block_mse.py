"""Block-wise MSE export using the unified experiment config.

Usage:
    python scripts/export_block_mse.py
    python scripts/export_block_mse.py --model resnet18
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate_blockwise_mse
from engine.utils import (
    build_cifar10_loaders,
    ensure_dir,
    get_model_output_dir,
    load_checkpoint,
    load_experiment_config,
    seed_everything,
    select_device,
)
from models import create_model, get_injection_points
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


def main():
    parser = argparse.ArgumentParser(description="Export block-wise MSE.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
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

    temperature = args.temperature or config["temperature"]["block_mse_temperature"]
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

    orig_mse = evaluate_blockwise_mse(
        model, test_loader, device, ideal_hooks, orig_hooks, temperature, seed)
    imp_mse = evaluate_blockwise_mse(
        model, test_loader, device, ideal_hooks, imp_hooks, temperature, seed)

    rows = []
    for name in OrderedDict.fromkeys(list(orig_mse.keys()) + list(imp_mse.keys())):
        o = orig_mse[name]
        i = imp_mse[name]
        reduction = (1 - i / o) * 100 if o > 0 else 0
        rows.append({
            "block": name,
            "temperature": temperature,
            "original_mse": o,
            "improved_mse": i,
            "reduction_pct": reduction,
        })

    frame = pd.DataFrame(rows)
    output_path = Path(args.output or str(out_dirs["csv"] / "blockwise_mse.csv"))
    ensure_dir(output_path.parent)
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
