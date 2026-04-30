"""一键运行完整实验流程。

适合第一次跑项目时使用。它会按固定顺序依次执行训练、评估和画图脚本。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(*args: str) -> None:
    """运行一个子脚本，并在失败时直接抛出异常。"""
    command = [sys.executable, *args]
    print("running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run_step("scripts/train_resnet18.py")
    run_step("scripts/eval_temperature.py")
    run_step("scripts/eval_noise.py")
    run_step("scripts/export_block_mse.py")
    run_step("scripts/eval_bin_sweep.py")
    run_step("scripts/plot_results.py")


if __name__ == "__main__":
    main()
