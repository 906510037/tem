# 运行说明

## 1. 环境准备

建议使用 Python 3.10 或更新版本，并在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

依赖包括：

- `torch`
- `torchvision`
- `PyYAML`
- `numpy`
- `pandas`
- `matplotlib`
- `pytest`

## 2. 推荐运行方式

最推荐使用 `run_experiment.py`，因为所有关键参数都集中在文件顶部的 `EXPERIMENT` 字典中，方便调参和复现实验。

完整运行训练、评估和画图：

```bash
python run_experiment.py
```

如果已经有 `outputs/checkpoints/best.pt`，只重新评估和画图：

```bash
python run_experiment.py --skip-train
```

运行结束后会生成：

```text
outputs/checkpoints/best.pt
outputs/csv/accuracy_vs_temp.csv
outputs/csv/accuracy_vs_noise.csv
outputs/csv/blockwise_mse.csv
outputs/csv/bin_sweep.csv
outputs/figures/*.png
outputs/figures/*.pdf
outputs/logs/
```

## 3. 分步骤运行方式

如果不想用一键脚本，也可以按顺序运行 `scripts/` 下的脚本。

训练 ResNet18：

```bash
python scripts/train_resnet18.py
```

温度扫描：

```bash
python scripts/eval_temperature.py
```

噪声扫描：

```bash
python scripts/eval_noise.py
```

导出逐块 MSE：

```bash
python scripts/export_block_mse.py
```

补偿分档扫描：

```bash
python scripts/eval_bin_sweep.py
```

生成图表：

```bash
python scripts/plot_results.py
```

旧的一键分步入口仍可使用：

```bash
python run_all.py
```

## 4. 重要配置位置

| 文件 | 作用 |
|---|---|
| `run_experiment.py` | 推荐主入口，顶部 `EXPERIMENT` 管理一键实验参数 |
| `configs/base.yaml` | 分步骤脚本的训练、评估、输出目录配置 |
| `configs/temp_original.yaml` | `original` 非理想参数 |
| `configs/temp_improved_2bin.yaml` | 2 档补偿配置 |
| `configs/temp_improved_4bin.yaml` | 4 档补偿配置，分步骤脚本默认使用 |
| `configs/temp_improved_6bin.yaml` | 6 档补偿配置 |

调参时优先改 `run_experiment.py` 顶部参数。如果使用 `scripts/` 分步骤运行，就同步修改 `configs/*.yaml`。

## 5. 数据集和 checkpoint

CIFAR-10 数据默认放在：

```text
data/
```

训练好的模型默认保存到：

```text
outputs/checkpoints/best.pt
```

如果运行 `--skip-train` 时提示找不到 checkpoint，说明需要先运行完整训练：

```bash
python run_experiment.py
```

## 6. 测试

运行单元测试：

```bash
pytest
```

测试主要覆盖：

- 指标计算
- 温度行为模型
- hook 注入逻辑

## 7. 常见问题

### 7.1 `outputs/csv/` 为空

说明当前还没有用默认输出目录跑过实验。运行：

```bash
python run_experiment.py
```

或已有 checkpoint 时运行：

```bash
python run_experiment.py --skip-train
```

### 7.2 `original` 高温下降太厉害

优先降低：

- `ka`
- `kb`
- `lam`

不要一开始就改 ResNet18 结构。只有参数已经很温和但仍然崩得太快时，再考虑减少注入的 residual block 数量。

### 7.3 `improved` 太接近 `ideal`

优先调整：

- 提高 `rho`
- 降低 `compensation_strength`
- 减少分档数量

### 7.4 `improved` 提升不明显

优先调整：

- 降低 `rho`
- 提高 `compensation_strength`
- 适当增加分档数量

### 7.5 分步骤脚本和 `run_experiment.py` 结果不同

这通常是因为两套入口的参数没有同步。`run_experiment.py` 使用顶部 `EXPERIMENT`，分步骤脚本使用 `configs/*.yaml`。正式实验前应固定一种入口，推荐固定使用 `run_experiment.py`。
