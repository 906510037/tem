"""Plot experiment results using the unified experiment config.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --model resnet18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from engine.utils import ensure_dir, get_model_output_dir, load_experiment_config


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accuracy_vs_temperature(frame: pd.DataFrame, figure_dir: Path, tag: str = "") -> None:
    title = f"Accuracy vs Temperature — {tag}" if tag else "Accuracy vs Temperature"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["ideal"], marker="o", linewidth=2, label="Ideal")
    ax.plot(frame["temperature"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["temperature"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7,
               label="High-temp boundary (85 C)")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "accuracy_vs_temperature")


def plot_accuracy_vs_noise(frame: pd.DataFrame, figure_dir: Path, tag: str = "") -> None:
    title = f"Accuracy vs Noise — {tag}" if tag else "Accuracy vs Noise"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["sigma"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["sigma"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.set_xlabel("sigma_0 (noise strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "accuracy_vs_noise")


def plot_block_mse(frame: pd.DataFrame, figure_dir: Path, tag: str = "") -> None:
    title = f"Block-wise Feature MSE — {tag}" if tag else "Block-wise Feature MSE"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(frame))
    ax.plot(x, frame["original_mse"], marker="s", linewidth=2, label="Original")
    ax.plot(x, frame["improved_mse"], marker="^", linewidth=2, label="Improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(frame["block"], rotation=45, ha="right")
    ax.set_xlabel("Block")
    ax.set_ylabel("Feature MSE")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "blockwise_feature_mse")


def plot_recovery_ratio(frame: pd.DataFrame, figure_dir: Path, tag: str = "") -> None:
    title = f"Accuracy Recovery Ratio — {tag}" if tag else "Accuracy Recovery Ratio"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(frame["temperature"], frame["recovery_ratio"], marker="o", linewidth=2,
            color="#2a6f97")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Recovery Ratio")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "accuracy_recovery_ratio")


def plot_bin_sweep(frame: pd.DataFrame, figure_dir: Path, tag: str = "") -> None:
    title = f"Compensation Bin Sweep — {tag}" if tag else "Compensation Bin Sweep"
    mean_frame = frame[frame["temperature"].astype(str) == "mean"].copy()
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
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "compensation_bin_sweep")

    detail = frame[frame["temperature"].astype(str) != "mean"].copy()
    if detail["temperature"].nunique() <= 1:
        return

    detail["temperature"] = detail["temperature"].astype(float)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for label in detail["bins"].unique():
        sub = detail[detail["bins"] == label]
        ax2.plot(sub["temperature"], sub["accuracy"], marker="o", linewidth=1.8,
                 label=f"{label}-bin")
    ax2.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax2.set_xlabel("Temperature (C)")
    ax2.set_ylabel("Accuracy")
    tag2 = f"Bin Sweep: Accuracy vs Temperature — {tag}" if tag else "Bin Sweep vs Temperature"
    ax2.set_title(tag2)
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    save_figure(fig2, figure_dir / "bin_sweep_vs_temperature")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or (ROOT / "configs" / "experiment.yaml")
    config = load_experiment_config(Path(config_path))
    model_name = args.model or config["models"]["active"][0]

    out_dirs = get_model_output_dir(config, model_name)
    csv_dir = out_dirs["csv"]
    figure_dir = out_dirs["figure"]
    ensure_dir(figure_dir)
    plt.style.use("default")

    temp_frame = pd.read_csv(csv_dir / "accuracy_vs_temp.csv")
    noise_frame = pd.read_csv(csv_dir / "accuracy_vs_noise.csv")
    mse_frame = pd.read_csv(csv_dir / "blockwise_mse.csv")
    bin_frame = pd.read_csv(csv_dir / "bin_sweep.csv")

    plot_accuracy_vs_temperature(temp_frame, figure_dir, model_name)
    plot_accuracy_vs_noise(noise_frame, figure_dir, model_name)
    plot_block_mse(mse_frame, figure_dir, model_name)
    plot_recovery_ratio(temp_frame, figure_dir, model_name)
    plot_bin_sweep(bin_frame, figure_dir, model_name)
    print(f"saved_figures: {figure_dir}")


if __name__ == "__main__":
    main()
