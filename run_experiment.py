"""一键运行完整实验 —— 所有参数集中在此文件修改。

使用方法:
    python run_experiment.py              # 运行全部实验（训练 + 评估 + 画图）
    python run_experiment.py --skip-train # 跳过训练，仅运行评估和画图
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.evaluate import evaluate_accuracy, evaluate_blockwise_mse
from engine.metrics import aggregate_feature_mse, recovery_ratio, update_accuracy_counts
from engine.train import fit
from engine.utils import (
    build_cifar10_loaders,
    ensure_dir,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    select_device,
)
from models import resnet18_cifar
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


# ============================================================
#  所有可调参数集中在此区域，修改这里即可控制整个实验
# ============================================================

EXPERIMENT = {

    "system": {
        "seed": 2026,
        "deterministic": True,
        "device": "auto",
    },

    "dataset": {
        "root": "./data",
        "num_classes": 10,
    },

    "model": {
        "base_channels": 64,
    },

    "train": {
        "epochs": 30,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "batch_size": 128,
        "num_workers": 2,
        "label_smoothing": 0.0,
    },

    "eval": {
        "batch_size": 256,
        "num_workers": 2,
    },

    "outputs": {
        "checkpoint_dir": "./outputs/checkpoints",
        "csv_dir": "./outputs/csv",
        "figure_dir": "./outputs/figures",
        "log_dir": "./outputs/logs",
    },

    # ----------------------------------------------------------
    #  温度评估点（0 ~ 135°C，>85°C 为高温）
    # ----------------------------------------------------------
    "temperature_points": [0, 25, 50, 75, 85, 95, 105, 115, 125, 135],

    # ----------------------------------------------------------
    #  噪声扫描参数
    # ----------------------------------------------------------
    "noise_scan": {
        "temperature": 75,
        "sigma_values": [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050],
    },

    # ----------------------------------------------------------
    #  Block-wise MSE 评估温度
    # ----------------------------------------------------------
    "block_mse_temperature": 105,

    # ----------------------------------------------------------
    #  Original 模式参数（无补偿）
    # ----------------------------------------------------------
    "original": {
        "mode": "original",
        "ka": 0.00048,
        "kb": 0.00018,
        "sigma0": 0.020,
        "lam": 0.006,
        "nominal_gain": 0.991,
        "nominal_bias": 0.0,
    },

    # ----------------------------------------------------------
    #  Improved 模式参数（有补偿）
    # ----------------------------------------------------------
    "improved": {
        "mode": "improved",
        "ka": 0.00048,
        "kb": 0.00018,
        "sigma0": 0.020,
        "lam": 0.006,
        "rho": 0.82,
        "nominal_gain": 0.991,
        "nominal_bias": 0.0,
        "compensation_strength": 0.72,
    },

    # ----------------------------------------------------------
    #  补偿档位定义（2-bin / 4-bin / 6-bin）
    # ----------------------------------------------------------
    "bin_configs": {
        "2": [0, 85, 135],
        "4": [0, 35, 65, 95, 135],
        "6": [0, 20, 40, 65, 85, 110, 135],
    },

    # ----------------------------------------------------------
    #  Bin Sweep 评估方式
    #  True  → 全温度范围平均（推荐，避免单点偏差）
    #  False → 仅在 bin_sweep_temperature 单点评估
    # ----------------------------------------------------------
    "bin_sweep_all_temps": True,
}


# ============================================================
#  以下为实验逻辑，一般无需修改
# ============================================================


def _build_base_config() -> Dict[str, Any]:
    return {
        "dataset": {**EXPERIMENT["dataset"], "name": "cifar10"},
        "model": {**EXPERIMENT["model"], "name": "resnet18_cifar"},
        "train": EXPERIMENT["train"],
        "eval": EXPERIMENT["eval"],
        "system": EXPERIMENT["system"],
        "outputs": EXPERIMENT["outputs"],
    }


def _make_original_config() -> NonIdealConfig:
    return NonIdealConfig.from_dict(EXPERIMENT["original"])


def _make_improved_config(bin_key: str = "4") -> NonIdealConfig:
    cfg = dict(EXPERIMENT["improved"])
    cfg["bin_edges"] = EXPERIMENT["bin_configs"][bin_key]
    return NonIdealConfig.from_dict(cfg)


def _make_hook_manager(
    blocks, config: NonIdealConfig, seed: int, mode: str = "replace_with_nonideal"
) -> NonIdealHookManager:
    return NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(config, noise_seed=seed),
        mode=mode,
    )


def step_train() -> Path:
    config = _build_base_config()
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
    print(f"[train] best_accuracy={result.best_accuracy:.4f}")
    return result.best_checkpoint


def step_temperature_scan(model: nn.Module, loader: DataLoader, device: torch.device, seed: int, csv_dir: Path):
    print("\n[1/4] Temperature scan ...")
    blocks = list(model.iter_residual_blocks())
    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    original_hooks = _make_hook_manager(blocks, _make_original_config(), seed)
    improved_hooks = _make_hook_manager(blocks, _make_improved_config("4"), seed)

    temperatures = EXPERIMENT["temperature_points"]
    rows = []
    for temp in temperatures:
        ideal_acc = evaluate_accuracy(model, loader, device, ideal_hooks, temp, seed)
        orig_acc = evaluate_accuracy(model, loader, device, original_hooks, temp, seed)
        imp_acc = evaluate_accuracy(model, loader, device, improved_hooks, temp, seed)
        rr = recovery_ratio(ideal_acc, orig_acc, imp_acc)
        rows.append({
            "temperature": temp,
            "ideal": ideal_acc,
            "original": orig_acc,
            "improved": imp_acc,
            "recovery_ratio": rr,
        })
        print(f"  T={temp:>5.0f}°C  Ideal={ideal_acc:.4f}  Orig={orig_acc:.4f}  Imp={imp_acc:.4f}  RR={rr:.4f}")

    frame = pd.DataFrame(rows)
    csv_path = csv_dir / "accuracy_vs_temp.csv"
    frame.to_csv(csv_path, index=False)
    return frame


def step_noise_scan(model: nn.Module, loader: DataLoader, device: torch.device, seed: int, csv_dir: Path):
    print("\n[2/4] Noise scan ...")
    blocks = list(model.iter_residual_blocks())
    original_hooks = _make_hook_manager(blocks, _make_original_config(), seed)
    improved_hooks = _make_hook_manager(blocks, _make_improved_config("4"), seed)

    noise_cfg = EXPERIMENT["noise_scan"]
    temperature = noise_cfg["temperature"]
    sigma_values = noise_cfg["sigma_values"]

    orig_module = original_hooks.nonideality
    imp_module = improved_hooks.nonideality
    orig_base_sigma = orig_module.config.sigma0
    imp_base_sigma = imp_module.config.sigma0

    rows = []
    try:
        for sigma0 in sigma_values:
            orig_module.config = type(orig_module.config)(**{**orig_module.config.__dict__, "sigma0": sigma0})
            imp_module.config = type(imp_module.config)(**{**imp_module.config.__dict__, "sigma0": sigma0})
            orig_acc = evaluate_accuracy(model, loader, device, original_hooks, temperature, seed)
            imp_acc = evaluate_accuracy(model, loader, device, improved_hooks, temperature, seed)
            rows.append({"sigma": sigma0, "original": orig_acc, "improved": imp_acc})
            print(f"  sigma0={sigma0:.3f}  Orig={orig_acc:.4f}  Imp={imp_acc:.4f}")
    finally:
        orig_module.config = type(orig_module.config)(**{**orig_module.config.__dict__, "sigma0": orig_base_sigma})
        imp_module.config = type(imp_module.config)(**{**imp_module.config.__dict__, "sigma0": imp_base_sigma})

    frame = pd.DataFrame(rows)
    csv_path = csv_dir / "accuracy_vs_noise.csv"
    frame.to_csv(csv_path, index=False)
    return frame


def step_block_mse(model: nn.Module, loader: DataLoader, device: torch.device, seed: int, csv_dir: Path):
    print("\n[3/4] Block-wise MSE ...")
    blocks = list(model.iter_residual_blocks())
    temperature = EXPERIMENT["block_mse_temperature"]

    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    original_hooks = _make_hook_manager(blocks, _make_original_config(), seed)
    improved_hooks = _make_hook_manager(blocks, _make_improved_config("4"), seed)

    orig_mse = evaluate_blockwise_mse(model, loader, device, ideal_hooks, original_hooks, temperature, seed)
    imp_mse = evaluate_blockwise_mse(model, loader, device, ideal_hooks, improved_hooks, temperature, seed)

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
        print(f"  {block_name:<12s}  Orig={o:.6f}  Imp={i:.6f}  ↓{reduction:.1f}%")

    frame = pd.DataFrame(rows)
    csv_path = csv_dir / "blockwise_mse.csv"
    frame.to_csv(csv_path, index=False)
    return frame


def step_bin_sweep(model: nn.Module, loader: DataLoader, device: torch.device, seed: int, csv_dir: Path):
    print("\n[4/4] Bin sweep ...")
    blocks = list(model.iter_residual_blocks())
    bin_configs = EXPERIMENT["bin_configs"]
    temperatures = EXPERIMENT["temperature_points"]
    all_temps = EXPERIMENT["bin_sweep_all_temps"]

    hook_managers = {}
    for bin_label, bin_edges in bin_configs.items():
        cfg = _make_improved_config(bin_label)
        hook_managers[bin_label] = _make_hook_manager(blocks, cfg, seed)

    eval_temps = temperatures if all_temps else [EXPERIMENT["temperature_points"][-3]]

    rows = []
    for bin_label, hooks in hook_managers.items():
        accs = []
        for temp in eval_temps:
            acc = evaluate_accuracy(model, loader, device, hooks, temp, seed)
            rows.append({"bins": bin_label, "temperature": temp, "accuracy": acc})
            accs.append(acc)
            print(f"  {bin_label}-bin  T={temp:>5.0f}°C  Acc={acc:.4f}")
        mean_acc = sum(accs) / len(accs)
        rows.append({"bins": bin_label, "temperature": "mean", "accuracy": mean_acc})
        print(f"  {bin_label}-bin  MEAN  Acc={mean_acc:.4f}")

    frame = pd.DataFrame(rows)
    csv_path = csv_dir / "bin_sweep.csv"
    frame.to_csv(csv_path, index=False)
    return frame


def step_plot(csv_dir: Path, figure_dir: Path):
    print("\n[plot] Generating figures ...")
    plt.style.use("default")
    ensure_dir(figure_dir)

    temp_frame = pd.read_csv(csv_dir / "accuracy_vs_temp.csv")
    noise_frame = pd.read_csv(csv_dir / "accuracy_vs_noise.csv")
    mse_frame = pd.read_csv(csv_dir / "blockwise_mse.csv")
    bin_frame = pd.read_csv(csv_dir / "bin_sweep.csv")

    _plot_accuracy_vs_temperature(temp_frame, figure_dir)
    _plot_accuracy_vs_noise(noise_frame, figure_dir)
    _plot_block_mse(mse_frame, figure_dir)
    _plot_recovery_ratio(temp_frame, figure_dir)
    _plot_bin_sweep(bin_frame, figure_dir)
    print(f"[plot] saved to {figure_dir}")


def _save_figure(fig: plt.Figure, path: Path):
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_accuracy_vs_temperature(frame: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["ideal"], marker="o", linewidth=2, label="Ideal")
    ax.plot(frame["temperature"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["temperature"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7, label="High-temp boundary (85°C)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Temperature")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "accuracy_vs_temperature")


def _plot_accuracy_vs_noise(frame: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["sigma"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["sigma"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.set_xlabel("σ₀ (noise strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Noise")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "accuracy_vs_noise")


def _plot_block_mse(frame: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(frame))
    ax.plot(x, frame["original_mse"], marker="s", linewidth=2, label="Original")
    ax.plot(x, frame["improved_mse"], marker="^", linewidth=2, label="Improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["block"], rotation=45, ha="right")
    ax.set_xlabel("Residual Block")
    ax.set_ylabel("Feature MSE")
    ax.set_title("Block-wise Feature MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir / "blockwise_feature_mse")


def _plot_recovery_ratio(frame: pd.DataFrame, figure_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["recovery_ratio"], marker="o", linewidth=2, color="#2a6f97")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Recovery Ratio")
    ax.set_title("Accuracy Recovery Ratio")
    ax.grid(alpha=0.25)
    _save_figure(fig, figure_dir / "accuracy_recovery_ratio")


def _plot_bin_sweep(frame: pd.DataFrame, figure_dir: Path):
    mean_frame = frame[frame["temperature"] == "mean"].copy()
    if len(mean_frame) == 0:
        mean_frame = frame.copy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]
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
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xlabel("Compensation Bins")
    ax.set_ylabel("Accuracy (mean over temperatures)")
    ax.set_title("Compensation Bin Sweep (full-range average)")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figure_dir / "compensation_bin_sweep")

    if "mean" not in frame["temperature"].astype(str).unique():
        return
    detail = frame[frame["temperature"] != "mean"].copy()
    detail["temperature"] = detail["temperature"].astype(float)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for bin_label in detail["bins"].unique():
        sub = detail[detail["bins"] == bin_label]
        ax2.plot(sub["temperature"], sub["accuracy"], marker="o", linewidth=1.8, label=f"{bin_label}-bin")
    ax2.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Bin Sweep: Accuracy vs Temperature")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    _save_figure(fig2, figure_dir / "bin_sweep_vs_temperature")


def main():
    parser = argparse.ArgumentParser(description="Run full experiment with centralized config.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing checkpoint.")
    args = parser.parse_args()

    config = _build_base_config()
    seed = config["system"]["seed"]
    seed_everything(seed, config["system"]["deterministic"])
    device = select_device(config["system"]["device"])

    csv_dir = ensure_dir(config["outputs"]["csv_dir"])
    figure_dir = ensure_dir(config["outputs"]["figure_dir"])

    if args.skip_train:
        checkpoint_path = Path(config["outputs"]["checkpoint_dir"]) / "best.pt"
        if not checkpoint_path.exists():
            print(f"[error] checkpoint not found: {checkpoint_path}")
            print("        run without --skip-train first, or check the path.")
            sys.exit(1)
    else:
        checkpoint_path = step_train()

    print(f"\n[load] checkpoint={checkpoint_path}")
    _, test_loader = build_cifar10_loaders(config, train=False)
    model = resnet18_cifar(
        num_classes=config["dataset"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    )
    load_checkpoint(checkpoint_path, model)
    model.to(device)

    step_temperature_scan(model, test_loader, device, seed, csv_dir)
    step_noise_scan(model, test_loader, device, seed, csv_dir)
    step_block_mse(model, test_loader, device, seed, csv_dir)
    step_bin_sweep(model, test_loader, device, seed, csv_dir)
    step_plot(csv_dir, figure_dir)

    print("\n✓ All experiments completed.")


if __name__ == "__main__":
    main()
