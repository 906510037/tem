"""一键运行完整实验 —— 所有参数在 configs/experiment.yaml。

使用方法:
    python run_experiment.py                          # 全流程
    python run_experiment.py --skip-train             # 跳过训练
    python run_experiment.py --models resnet18_cifar  # 单模型
    python run_experiment.py --epochs 10              # 快速验证
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
from tqdm import tqdm

from engine.evaluate import evaluate_accuracy, evaluate_blockwise_mse
from engine.metrics import recovery_ratio
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
from nonideal.hooks import NonIdealHookManager
from nonideal.temp_model import NonIdealConfig, TemperatureNonIdeality


# ============================================================
#  配置构建
# ============================================================

def _make_nonideal_config(params: Dict[str, Any],
                          config: Dict[str, Any] | None = None) -> NonIdealConfig:
    """Build NonIdealConfig, with default 4-bin edges for improved mode."""
    p = dict(params)
    if p.get("mode") == "improved" and "bin_edges" not in p and config is not None:
        p["bin_edges"] = config["nonideal"].get("bins", {}).get(
            "4", [0, 28, 55, 83, 110])
    return NonIdealConfig.from_dict(p)


def _make_hook_manager(
    blocks: list, ncfg: NonIdealConfig, seed: int,
    mode: str = "replace_with_nonideal",
) -> NonIdealHookManager:
    return NonIdealHookManager(
        blocks,
        TemperatureNonIdeality(ncfg, noise_seed=seed),
        mode=mode,
    )


# ============================================================
#  训练
# ============================================================

def step_train(config: Dict[str, Any], model_name: str,
               skip: bool = False) -> Path | None:
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
    print(f"\n[1/2] Temperature scan — {model_name} ...")
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    blocks = get_injection_points(model, arch)
    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    orig_hooks = _make_hook_manager(
        blocks, _make_nonideal_config(config["nonideal"]["original"]), seed)
    imp_hooks = _make_hook_manager(
        blocks, _make_nonideal_config(config["nonideal"]["improved"], config), seed)

    temperatures = get_temperature_points(config)
    rows = []
    for temp in tqdm(temperatures, desc=f"Temp scan {model_name}", leave=False):
        ideal_acc = evaluate_accuracy(model, loader, device, ideal_hooks, temp, seed)
        orig_acc = evaluate_accuracy(model, loader, device, orig_hooks, temp, seed)
        imp_acc = evaluate_accuracy(model, loader, device, imp_hooks, temp, seed)
        rr = recovery_ratio(ideal_acc, orig_acc, imp_acc)
        rows.append({
            "temperature": temp, "ideal": ideal_acc,
            "original": orig_acc, "improved": imp_acc, "recovery_ratio": rr,
        })

    frame = pd.DataFrame(rows)
    out_dirs = get_model_output_dir(config, model_name)
    csv_path = out_dirs["csv"] / "accuracy_vs_temp.csv"
    frame.to_csv(csv_path, index=False)

    # Summary at key points
    key_temps = [0, 25, 50, 75, 85, 100, 110]
    for t_key in key_temps:
        idx = min(range(len(temperatures)), key=lambda i: abs(temperatures[i] - t_key))
        row = frame.iloc[idx]
        print(f"  T={row['temperature']:>5.0f}C  Ideal={row['ideal']:.4f}  "
              f"Orig={row['original']:.4f}  Imp={row['improved']:.4f}  "
              f"RR={row['recovery_ratio']:.4f}")
    return frame


# ============================================================
#  逐块 MSE
# ============================================================

def step_block_mse(
    model: nn.Module, loader: DataLoader, device: torch.device,
    seed: int, config: Dict[str, Any], model_name: str,
) -> pd.DataFrame:
    print(f"\n[2/2] Block-wise MSE — {model_name} ...")
    model_cfg = config["models"][model_name]
    arch = model_cfg["arch"]
    blocks = get_injection_points(model, arch)
    temperature = config["temperature"]["block_mse_temperature"]

    ideal_hooks = NonIdealHookManager(blocks, nonideality=None, mode="capture_only")
    orig_hooks = _make_hook_manager(
        blocks, _make_nonideal_config(config["nonideal"]["original"]), seed)
    imp_hooks = _make_hook_manager(
        blocks, _make_nonideal_config(config["nonideal"]["improved"], config), seed)

    orig_mse = evaluate_blockwise_mse(
        model, loader, device, ideal_hooks, orig_hooks, temperature, seed)
    imp_mse = evaluate_blockwise_mse(
        model, loader, device, ideal_hooks, imp_hooks, temperature, seed)

    rows = []
    for block_name in OrderedDict.fromkeys(list(orig_mse.keys()) + list(imp_mse.keys())):
        o = orig_mse[block_name]
        i = imp_mse[block_name]
        reduction = (1 - i / o) * 100 if o > 0 else 0
        rows.append({"block": block_name, "temperature": temperature,
                     "original_mse": o, "improved_mse": i, "reduction_pct": reduction})
        print(f"  {block_name:<20s}  Orig={o:.6f}  Imp={i:.6f}  d={reduction:.1f}%")

    frame = pd.DataFrame(rows)
    out_dirs = get_model_output_dir(config, model_name)
    csv_path = out_dirs["csv"] / "blockwise_mse.csv"
    frame.to_csv(csv_path, index=False)
    return frame


# ============================================================
#  单模型画图
# ============================================================

def step_plot(config: Dict[str, Any], model_name: str):
    out_dirs = get_model_output_dir(config, model_name)
    csv_dir = out_dirs["csv"]
    figure_dir = out_dirs["figure"]
    print(f"\n[plot] {model_name} ...")
    plt.style.use("default")
    ensure_dir(figure_dir)

    temp_frame = pd.read_csv(csv_dir / "accuracy_vs_temp.csv")
    mse_frame = pd.read_csv(csv_dir / "blockwise_mse.csv")

    _plot_accuracy_vs_temperature(temp_frame, figure_dir, model_name)
    _plot_recovery_ratio(temp_frame, figure_dir, model_name)
    _plot_block_mse(mse_frame, figure_dir, model_name)
    print(f"[plot] {model_name}: saved to {figure_dir}")


# ============================================================
#  交叉模型对比图 + 汇总表
# ============================================================

def step_cross_model(config: Dict[str, Any], model_names: list[str]):
    if len(model_names) < 2:
        return
    print(f"\n[plot] Cross-model comparison ({', '.join(model_names)}) ...")

    root = Path(config["outputs"]["root"])
    cross_dir = ensure_dir(root / "cross_model")
    plt.style.use("default")

    # ---- 汇总温度扫描 ----
    all_temp: dict[str, pd.DataFrame] = {}
    for m in model_names:
        p = root / m / "csv" / "accuracy_vs_temp.csv"
        if p.exists():
            all_temp[m] = pd.read_csv(p)

    if all_temp:
        _plot_cross_temperature(all_temp, cross_dir)
        _save_summary_table(all_temp, cross_dir)

    # ---- 汇总 MSE ----
    all_mse: dict[str, pd.DataFrame] = {}
    for m in model_names:
        p = root / m / "csv" / "blockwise_mse.csv"
        if p.exists():
            all_mse[m] = pd.read_csv(p)

    if all_mse:
        _plot_cross_mse(all_mse, cross_dir)
        _save_mse_table(all_mse, cross_dir)

    print(f"[plot] cross-model: saved to {cross_dir}")


# ---- 交叉模型图 ----

def _plot_cross_temperature(all_temp: dict[str, pd.DataFrame], cross_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2a6f97", "#d95f02", "#7570b3"]
    markers = ["o", "s", "^"]
    for (name, df), c, m in zip(all_temp.items(), colors, markers):
        ax1.plot(df["temperature"], df["original"], marker=m, linewidth=2,
                 color=c, label=f"{name} Original")
        ax2.plot(df["temperature"], df["improved"], marker=m, linewidth=2,
                 color=c, label=f"{name} Improved")

    for ax in (ax1, ax2):
        ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    ax1.set_title("Original Accuracy vs Temperature")
    ax2.set_title("Improved Accuracy vs Temperature")
    fig.suptitle("Cross-Model Comparison — Accuracy vs Temperature", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, cross_dir / "cross_accuracy_vs_temperature")


def _plot_cross_mse(all_mse: dict[str, pd.DataFrame], cross_dir: Path):
    # Bar chart: each model's final-block MSE comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(all_mse.keys())
    x = np.arange(len(models))
    width = 0.35
    orig_vals = [all_mse[m]["original_mse"].iloc[-1] for m in models]
    imp_vals = [all_mse[m]["improved_mse"].iloc[-1] for m in models]

    ax.bar(x - width / 2, orig_vals, width, label="Original", color="#d95f02")
    ax.bar(x + width / 2, imp_vals, width, label="Improved", color="#2a6f97")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Feature MSE (deepest block)")
    ax.set_title("Cross-Model Block-wise MSE Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_fig(fig, cross_dir / "cross_blockwise_mse")


# ---- 汇总 CSV ----

def _save_summary_table(all_temp: dict[str, pd.DataFrame], cross_dir: Path):
    key_temps = [0, 25, 50, 75, 85, 100, 110]
    rows = []
    for name, df in all_temp.items():
        for tk in key_temps:
            temps = df["temperature"].values
            idx = min(range(len(temps)), key=lambda i: abs(temps[i] - tk))
            row = df.iloc[idx]
            rows.append({
                "model": name, "temperature": int(tk),
                "ideal": round(row["ideal"], 4),
                "original": round(row["original"], 4),
                "improved": round(row["improved"], 4),
                "recovery_ratio": round(row["recovery_ratio"], 4),
            })
    pd.DataFrame(rows).to_csv(cross_dir / "cross_model_summary.csv", index=False)


def _save_mse_table(all_mse: dict[str, pd.DataFrame], cross_dir: Path):
    rows = []
    for name, df in all_mse.items():
        for _, r in df.iterrows():
            rows.append({"model": name, "block": r["block"],
                         "original_mse": r["original_mse"],
                         "improved_mse": r["improved_mse"],
                         "reduction_pct": r["reduction_pct"]})
    pd.DataFrame(rows).to_csv(cross_dir / "cross_model_mse.csv", index=False)


# ---- 单模型绘图 ----

def _save_fig(fig: plt.Figure, path: Path):
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
               label="High-temp (85 C)")
    ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Temperature — {tag}")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    _save_fig(fig, figure_dir / "accuracy_vs_temperature")


def _plot_recovery_ratio(frame: pd.DataFrame, figure_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["recovery_ratio"], marker="o", linewidth=2,
            color="#2a6f97")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax.set_xlabel("Temperature (C)"); ax.set_ylabel("Recovery Ratio")
    ax.set_title(f"Recovery Ratio — {tag}")
    ax.grid(alpha=0.25)
    _save_fig(fig, figure_dir / "accuracy_recovery_ratio")


def _plot_block_mse(frame: pd.DataFrame, figure_dir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(frame))
    ax.plot(x, frame["original_mse"], marker="s", linewidth=2, label="Original")
    ax.plot(x, frame["improved_mse"], marker="^", linewidth=2, label="Improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["block"], rotation=45, ha="right")
    ax.set_xlabel("Block"); ax.set_ylabel("Feature MSE")
    ax.set_title(f"Block-wise MSE — {tag}")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    _save_fig(fig, figure_dir / "blockwise_feature_mse")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run full experiment.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or (ROOT / "configs" / "experiment.yaml")
    config = load_experiment_config(Path(config_path))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs

    active_models: list[str] = (
        args.models.split(",") if args.models else config["models"]["active"]
    )

    system_cfg = config["system"]
    seed_everything(system_cfg["seed"], system_cfg["deterministic"])
    device = select_device(system_cfg["device"])

    temps = get_temperature_points(config)
    print(f"Temperature: {temps[0]}–{temps[-1]} C, {len(temps)} pts, "
          f"step {config['temperature']['step']} C")
    print(f"Models: {active_models}")
    print(f"Non-Ideal: ka={config['nonideal']['original']['ka']}, "
          f"nom_gain={config['nonideal']['original']['nominal_gain']}, "
          f"sigma0={config['nonideal']['original']['sigma0']}")
    print(f"  Improved: rho={config['nonideal']['improved']['rho']}, "
          f"comp={config['nonideal']['improved']['compensation_strength']}")

    for model_name in active_models:
        print(f"\n{'='*55}\n  MODEL: {model_name}\n{'='*55}")

        checkpoint_path = step_train(config, model_name, skip=args.skip_train)
        if checkpoint_path is None:
            print(f"[error] No checkpoint for {model_name}, skipping.")
            continue

        print(f"\n[load] {model_name}: {checkpoint_path}")
        _, test_loader = build_cifar10_loaders(config, train=False)
        model = create_model(
            model_name, config["models"][model_name],
            num_classes=config["dataset"]["num_classes"],
        )
        load_checkpoint(checkpoint_path, model)
        model.to(device)

        seed = system_cfg["seed"]
        step_temperature_scan(model, test_loader, device, seed, config, model_name)
        step_block_mse(model, test_loader, device, seed, config, model_name)
        step_plot(config, model_name)

    step_cross_model(config, active_models)
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
