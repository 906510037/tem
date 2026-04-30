from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from engine.utils import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--temp-csv", default="outputs/csv/accuracy_vs_temp.csv")
    parser.add_argument("--noise-csv", default="outputs/csv/accuracy_vs_noise.csv")
    parser.add_argument("--mse-csv", default="outputs/csv/blockwise_mse.csv")
    parser.add_argument("--bin-csv", default="outputs/csv/bin_sweep.csv")
    parser.add_argument("--figure-dir", default=None)
    return parser.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_accuracy_vs_temperature(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(frame["temperature"], frame["ideal"], marker="o", linewidth=2, label="Ideal")
    ax.plot(frame["temperature"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["temperature"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7, label="High-temp boundary")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Temperature")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "accuracy_vs_temperature")


def plot_accuracy_vs_noise(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(frame["sigma"], frame["original"], marker="s", linewidth=2, label="Original")
    ax.plot(frame["sigma"], frame["improved"], marker="^", linewidth=2, label="Improved")
    ax.set_xlabel("Sigma0")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Noise")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "accuracy_vs_noise")


def plot_block_mse(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
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
    save_figure(fig, figure_dir / "blockwise_feature_mse")


def plot_recovery_ratio(frame: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(frame["temperature"], frame["recovery_ratio"], marker="o", linewidth=2, color="#2a6f97")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Recovery Ratio")
    ax.set_title("Accuracy Recovery Ratio")
    ax.grid(alpha=0.25)
    save_figure(fig, figure_dir / "accuracy_recovery_ratio")


def plot_bin_sweep(frame: pd.DataFrame, figure_dir: Path) -> None:
    mean_frame = frame[frame["temperature"].astype(str) == "mean"].copy()
    if len(mean_frame) == 0:
        mean_frame = frame.copy()

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    colors = ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"]
    bars = ax.bar(mean_frame["bins"].astype(str), mean_frame["accuracy"], color=colors[: len(mean_frame)])
    for bar, accuracy in zip(bars, mean_frame["accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{accuracy:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xlabel("Compensation Bins")
    ax.set_ylabel("Accuracy")
    ax.set_title("Compensation Bin Sweep")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figure_dir / "compensation_bin_sweep")

    detail = frame[frame["temperature"].astype(str) != "mean"].copy()
    if detail["temperature"].nunique() <= 1:
        return

    detail["temperature"] = detail["temperature"].astype(float)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label in detail["bins"].astype(str).unique():
        sub = detail[detail["bins"].astype(str) == label]
        ax.plot(sub["temperature"], sub["accuracy"], marker="o", linewidth=1.8, label=f"{label}-bin")
    ax.axvline(85, linestyle="--", linewidth=1.2, color="#6c757d", alpha=0.7)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Bin Sweep vs Temperature")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, figure_dir / "bin_sweep_vs_temperature")


def main() -> None:
    args = parse_args()
    base_config = load_yaml(args.base_config)
    plt.style.use("default")

    figure_dir = Path(args.figure_dir or base_config["outputs"]["figure_dir"])
    ensure_dir(figure_dir)

    temp_frame = pd.read_csv(args.temp_csv)
    noise_frame = pd.read_csv(args.noise_csv)
    mse_frame = pd.read_csv(args.mse_csv)
    bin_frame = pd.read_csv(args.bin_csv)

    plot_accuracy_vs_temperature(temp_frame, figure_dir)
    plot_accuracy_vs_noise(noise_frame, figure_dir)
    plot_block_mse(mse_frame, figure_dir)
    plot_recovery_ratio(temp_frame, figure_dir)
    plot_bin_sweep(bin_frame, figure_dir)
    print(f"saved_figures={figure_dir}")


if __name__ == "__main__":
    main()
