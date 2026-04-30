"""一键运行完整实验 —— 所有参数集中在 configs/experiment.yaml。

使用方法:
    python run_experiment.py                  # 运行全部实验（训练 + 评估 + 画图）
    python run_experiment.py --skip-train     # 跳过训练，仅运行评估和画图
    python run_experiment.py --models resnet18  # 仅运行指定模型
    python run_experiment.py --epochs 5       # 快速验证（覆盖 YAML 中 epoch 数）
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.evaluate import evaluate_accuracy, evaluate_blockwise_mse
from engine.metrics import aggregate_feature_mse, recovery_ratio
from engine.train import fit
from engine.utils import (
    build_cifar10_loaders,
    ensure_dir,
    get_model_output_dir,
    get_temperature_points,
    load_checkpoint,
    load_experiment_config,
    seed_everything,
    select_device,
)
from models import create_model, get_injection_points
from nonideal.hooks import NonIdealHookManager, hook_session
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


# ============================================================
#  配置构建
# ============================================================

def _make_nonideal_config(params: Dict[str, Any]) -> NonIdealConfig:
    return NonIdealConfig.from_dict(params)


def _make_hook_manager(
    blocks: list, config: NonIdealConfig, seed: int,
    mode: str = "replace_with_nonideal",
) -> NonIdealHookManager:
    return NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(config, noise_seed=seed),
        mode=mode,
    )


# ============================================================
#  训练
# ============================================================

def step_train(config: Dict[str, Any], model_name: str, skip: bool = False
               ) -> Path | None:
    out_dirs = get_model_output_dir(config, model_name)
    checkpoint_path = out_dirs["checkpoint"] / "best.pt"

    if skip and checkpoint_path.exists():
        print(f"[train] {model_name}: using cached checkpoint {checkpoint_path}")
        return checkpoint_path

    system_cfg = config["system"]
    seed_everything(system_cfg["seed"], system_cfg["deterministic"])
    device = select_device(system_cfg["device"])

    train_loader, test_loader = build_cifar10_loaders(config, train=True)
    model_cfg = config["models"][model_name]
    model = create_model(model_name, model_cfg,
                         num_classes=config["dataset"]["num_classes"])
    result = fit(model, train_loader, test_loader, config, device,
                 model_name=model_name)
    print(f"[train] {model_name}: best_accuracy={result.best_accuracy:.4f}")
    return result.best_checkpoint


# ============================================================
#  温度扫描
# ============================================================

def step_temperature_scan(
    model: nn.Module, loader: DataLoader, device: torch.device,
    seed: int, config: Dict[str, Any], model_name: str,
) -> pd.DataFrame:
    print(f"\n[1/4] Temperature scan — {model_name} ...")
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    blocks = get_injection_points(model, arch)

    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    orig_hooks = _make_hook_manager(blocks, _make_nonideal_config(config["nonideal"]["original"]), seed)
    imp_hooks = _make_hook_manager(blocks, _make_nonideal_config(config["nonideal"]["improved"]), seed)

    temperatures = get_temperature_points(config)
    rows = []
    for temp in temperatures:
        ideal_acc = evaluate_accuracy(model, loader, device, ideal_hooks, temp, seed)
        orig_acc = evaluate_accuracy(model, loader, device, orig_hooks, temp, seed)
        imp_acc = evaluate_accuracy(model, loader, device, imp_hooks, temp, seed)
        rr = recovery_ratio(ideal_acc, orig_acc, imp_acc)
        rows.append({
            "temperature": temp,
            "ideal": ideal_acc,
            "original": orig_acc,
            "improved": imp_acc,
            "recovery_ratio": rr,
        })
        if int(temp) % 20 == 0 or temp == temperatures[-1]:
            print(f"  T={temp:>5.0f}C  Ideal={ideal_acc:.4f}  "
                  f"Orig={orig_acc:.4f}  Imp={imp_acc:.4f}  RR={rr:.4f}")

    frame = pd.DataFrame(rows)
    out_dirs = get_model_output_dir(config, model_name)
    csv_path = out_dirs["csv"] / "accuracy_vs_temp.csv"
    frame.to_csv(csv_path, index=False)
    return frame


# ============================================================
#  噪声扫描
# ============================================================

def step_noise_scan(
    model: nn.Module, loader: DataLoader, device: torch.device,
    seed: int, config: Dict[str, Any], model_name: str,
) -> pd.DataFrame:
    print(f"\n[2/4] Noise scan — {model_name} ...")
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    blocks = get_injection_points(model, arch)

    orig_hooks = _make_hook_manager(blocks, _make_nonideal_config(config["nonideal"]["original"]), seed)
    imp_hooks = _make_hook_manager(blocks, _make_nonideal_config(config["nonideal"]["improved"]), seed)

    tcfg = config["temperature"]
    temperature = tcfg["noise_temperature"]
    sigma_values = tcfg["noise_scan_values"]

    orig_module = orig_hooks.nonideality
    imp_module = imp_hooks.nonideality
    orig_base_sigma = orig_module.config.sigma0
    imp_base_sigma = imp_module.config.sigma0

    rows = []
    try:
        for sigma0 in sigma_values:
            orig_module.config = type(orig_module.config)(
                **{**orig_module.config.__dict__, "sigma0": sigma0})
            imp_module.config = type(imp_module.config)(
                **{**imp_module.config.__dict__, "sigma0": sigma0})
            orig_acc = evaluate_accuracy(model, loader, device, orig_hooks, temperature, seed)
            imp_acc = evaluate_accuracy(model, loader, device, imp_hooks, temperature, seed)
            rows.append({"sigma": sigma0, "original": orig_acc, "improved": imp_acc})
            print(f"  sigma0={sigma0:.3f}  Orig={orig_acc:.4f}  Imp={imp_acc:.4f}")
    finally:
        orig_module.config = type(orig_module.config)(
            **{**orig_module.config.__dict__, "sigma0": orig_base_sigma})
        imp_module.config = type(imp_module.config)(
            **{**imp_module.config.__dict__, "sigma0": imp_base_sigma})

    frame = pd.DataFrame(rows)
    out_dirs = get_model_output_dir(config, model_name)
    csv_path = out_dirs["csv"] / "accuracy_vs_noise.csv"
    frame.to_csv(csv_path, index=False)
    return frame


# ============================================================
#  逐块 MSE
# ============================================================

def step_block_mse(
    model: nn.Module, loader: DataLoader, device: torch.device,
    seed: int, config: Dict[str, Any], model_name: str,
) -> pd.DataFrame:
    print(f"\n[3/4] Block-wise MSE — {model_name} ...")
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    blocks = get_injection_points(model, arch)

    temperature = config["temperature"]["block_mse_temperature"]

    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    orig_hooks = _make_hook_manager(blocks, _make_nonideal_config(config["nonideal"]["original"]), seed)
    imp_hooks = _make_hook_manager(blocks, _make_nonideal_config(config["nonideal"]["improved"]), seed)

    orig_mse = evaluate_blockwise_mse(model, loader, device, ideal_hooks, orig_hooks, temperature, seed)
    imp_mse = evaluate_blockwise_mse(model, loader, device, ideal_hooks, imp_hooks, temperature, seed)

    rows = []
    for block_name in OrderedDict.fromkeys(list(orig_mse.keys()) + list(imp_mse.keys())):
        o = orig_mse[block_name]
        i = imp_mse[block_name]
        reduction = (1 - i / o) * 100 if o > 0 else 0
        rows.append({
            "block": block_name,
            "temperature": temperature,
            "original_mse": o,
            "improved_mse": i,
            "reduction_pct": reduction,
        })
        print(f"  {block_name:<20s}  Orig={o:.6f}  Imp={i:.6f}  d={reduction:.1f}%")

    frame = pd.DataFrame(rows)
    out_dirs = get_model_output_dir(config, model_name)
    csv_path = out_dirs["csv"] / "blockwise_mse.csv"
    frame.to_csv(csv_path, index=False)
    return frame


# ============================================================
#  分档扫描
# ============================================================

def step_bin_sweep(
    model: nn.Module, loader: DataLoader, device: torch.device,
    seed: int, config: Dict[str, Any], model_name: str,
) -> pd.DataFrame:
    print(f"\n[4/4] Bin sweep — {model_name} ...")
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    blocks = get_injection_points(model, arch)

    bin_configs = config["nonideal"]["bins"]
    active_sweep = bin_configs.get("active_sweep", list(bin_configs.keys()))
    temperatures = get_temperature_points(config)
    all_temps = config["temperature"]["bin_sweep_all_temps"]

    hook_managers = {}
    for bin_label in active_sweep:
        imp = dict(config["nonideal"]["improved"])
        imp["bin_edges"] = bin_configs[bin_label]
        hook_managers[bin_label] = _make_hook_manager(
            blocks, _make_nonideal_config(imp), seed)

    eval_temps = temperatures if all_temps else [config["temperature"]["block_mse_temperature"]]

    rows = []
    for bin_label, hooks in hook_managers.items():
        accs = []
        for temp in eval_temps:
            acc = evaluate_accuracy(model, loader, device, hooks, temp, seed)
            rows.append({"bins": bin_label, "temperature": temp, "accuracy": acc})
            accs.append(acc)
        mean_acc = sum(accs) / len(accs)
        rows.append({"bins": bin_label, "temperature": "mean", "accuracy": mean_acc})
        print(f"  {bin_label}-bin  MEAN  Acc={mean_acc:.4f}")

    frame = pd.DataFrame(rows)
    out_dirs = get_model_output_dir(config, model_name)
    csv_path = out_dirs["csv"] / "bin_sweep.csv"
    frame.to_csv(csv_path, index=False)
    return frame


# ============================================================
#  画图
# ============================================================

def step_plot(config: Dict[str, Any], model_name: str):
    out_dirs = get_model_output_dir(config, model_name)
    csv_dir = out_dirs["csv"]
    figure_dir = out_dirs["figure"]

    print(f"\n[plot] Generating figures — {model_name} ...")
    plt.style.use("default")
    ensure_dir(figure_dir)

    temp_frame = pd.read_csv(csv_dir / "accuracy_vs_temp.csv")
    noise_frame = pd.read_csv(csv_dir / "accuracy_vs_noise.csv")
    mse_frame = pd.read_csv(csv_dir / "blockwise_mse.csv")
    bin_frame = pd.read_csv(csv_dir / "bin_sweep.csv")

    _plot_accuracy_vs_temperature(temp_frame, figure_dir, model_name)
    _plot_accuracy_vs_noise(noise_frame, figure_dir, model_name)
    _plot_block_mse(mse_frame, figure_dir, model_name)
    _plot_recovery_ratio(temp_frame, figure_dir, model_name)
    _plot_bin_sweep(bin_frame, figure_dir, model_name)
    print(f"[plot] {model_name}: saved to {figure_dir}")


def _save_figure(fig: plt.Figure, path: Path):
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_accuracy_vs_temperature(frame: pd.DataFrame, figure_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["ideal"], marker="o", linewidth=2, label="Ideal")
    ax.plot(frame["temperature"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["temperature"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7,
               label="High-temp boundary (85 C)")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Temperature — {tag}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "accuracy_vs_temperature")


def _plot_accuracy_vs_noise(frame: pd.DataFrame, figure_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["sigma"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["sigma"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.set_xlabel("sigma_0 (noise strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Noise — {tag}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "accuracy_vs_noise")


def _plot_block_mse(frame: pd.DataFrame, figure_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(frame))
    ax.plot(x, frame["original_mse"], marker="s", linewidth=2, label="Original")
    ax.plot(x, frame["improved_mse"], marker="^", linewidth=2, label="Improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["block"], rotation=45, ha="right")
    ax.set_xlabel("Block")
    ax.set_ylabel("Feature MSE")
    ax.set_title(f"Block-wise Feature MSE — {tag}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "blockwise_feature_mse")


def _plot_recovery_ratio(frame: pd.DataFrame, figure_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["recovery_ratio"], marker="o", linewidth=2,
            color="#2a6f97")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Recovery Ratio")
    ax.set_title(f"Accuracy Recovery Ratio — {tag}")
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir / "accuracy_recovery_ratio")


def _plot_bin_sweep(frame: pd.DataFrame, figure_dir: Path, tag: str):
    mean_frame = frame[frame["temperature"] == "mean"].copy()
    if len(mean_frame) == 0:
        mean_frame = frame.copy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500", "#a96b2d"]
    bars = ax.bar(
        mean_frame["bins"].astype(str),
        mean_frame["accuracy"],
        color=colors[: len(mean_frame)],
    )
    for bar, acc in zip(bars, mean_frame["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.2%}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_xlabel("Compensation Bins")
    ax.set_ylabel("Accuracy (mean over temperatures)")
    ax.set_title(f"Compensation Bin Sweep — {tag}")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_dir / "compensation_bin_sweep")

    if "mean" not in frame["temperature"].astype(str).unique():
        return
    detail = frame[frame["temperature"] != "mean"].copy()
    detail["temperature"] = detail["temperature"].astype(float)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for bin_label in detail["bins"].unique():
        sub = detail[detail["bins"] == bin_label]
        ax2.plot(sub["temperature"], sub["accuracy"], marker="o", linewidth=1.8,
                 label=f"{bin_label}-bin")
    ax2.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax2.set_xlabel("Temperature (C)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Bin Sweep: Accuracy vs Temperature — {tag}")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    _save_figure(fig2, figure_dir / "bin_sweep_vs_temperature")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run full experiment with unified config.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing checkpoint.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (overrides config).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs (for quick validation).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment YAML config.")
    args = parser.parse_args()

    config_path = args.config or (ROOT / "configs" / "experiment.yaml")
    config = load_experiment_config(Path(config_path))

    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    active_models: list[str] = args.models.split(",") if args.models else config["models"]["active"]

    system_cfg = config["system"]
    seed_everything(system_cfg["seed"], system_cfg["deterministic"])
    device = select_device(system_cfg["device"])

    temperatures = get_temperature_points(config)
    print(f"Temperature range: {temperatures[0]} – {temperatures[-1]} C "
          f"({len(temperatures)} points, step {config['temperature']['step']} C)")
    print(f"Models: {active_models}")

    for model_name in active_models:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        checkpoint_path = step_train(config, model_name, skip=args.skip_train)
        if checkpoint_path is None:
            print(f"[error] No checkpoint for {model_name}, skipping evaluation.")
            continue

        print(f"\n[load] {model_name}: checkpoint={checkpoint_path}")
        _, test_loader = build_cifar10_loaders(config, train=False)
        model = create_model(
            model_name,
            config["models"][model_name],
            num_classes=config["dataset"]["num_classes"],
        )
        load_checkpoint(checkpoint_path, model)
        model.to(device)

        seed = system_cfg["seed"]
        step_temperature_scan(model, test_loader, device, seed, config, model_name)
        step_noise_scan(model, test_loader, device, seed, config, model_name)
        step_block_mse(model, test_loader, device, seed, config, model_name)
        step_bin_sweep(model, test_loader, device, seed, config, model_name)
        step_plot(config, model_name)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
