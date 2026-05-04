# 运行说明

## 1. 环境准备

Python 3.9+，安装依赖：

```bash
pip install -r requirements.txt
```

| 依赖 | 用途 |
|---|---|
| `torch` `torchvision` | 模型训练和推理 |
| `PyYAML` | 配置文件解析 |
| `numpy` `pandas` | 数据处理 |
| `matplotlib` | 画图 |
| `pytest` | 测试 |

## 2. 快速开始

所有参数统一由 `configs/experiment.yaml` 管理，**不再需要手动同步多个配置文件**。

### 一键运行

```bash
# 训练 + 评估 + 画图（默认 4 个模型）
python run_experiment.py

# 单模型（推荐用于快速验证）
python run_experiment.py --models resnet18_cifar

# GPU 环境
python run_experiment.py --models resnet18_cifar --epochs 30

# 服务器后台运行
nohup python run_experiment.py --models resnet18_cifar --epochs 30 > experiment.log 2>&1 &
```

### 跳过训练（已有 checkpoint）

```bash
python run_experiment.py --skip-train
```

### 命令行覆盖

```bash
python run_experiment.py --models resnet18_cifar,resnet34_cifar --epochs 30
```

### 输出目录结构

```text
outputs/
  resnet18_cifar/
    checkpoints/          # best.pt + top-3 checkpoints
    csv/                  # accuracy_vs_temp.csv, accuracy_vs_noise.csv, blockwise_mse.csv, bin_sweep.csv
    figures/              # *.png + *.pdf
    log/                  # train.log
  resnet34_cifar/
  vgg16_cifar/
  mobilenetv2_cifar/
```

## 3. 输出目录

```text
outputs/
  resnet18_cifar/
    checkpoints/  csv/  figures/  log/
  resnet34_cifar/
  resnet50_cifar/
  cross_model/           # 三模型交叉对比图表
```

## 4. 分步骤运行

```bash
python scripts/train_resnet18.py --model resnet18_cifar
python scripts/eval_temperature.py --model resnet18_cifar
python scripts/export_block_mse.py --model resnet18_cifar
python scripts/plot_results.py --model resnet18_cifar
```

## 4. 配置

单文件 `configs/experiment.yaml`，全部参数集中在此：

```yaml
system:         # seed, device
dataset:        # CIFAR-10 路径、类别数
models:         # 模型列表和各自参数
train:          # epochs, lr, batch_size, warmup, cutout ...
eval:           # 评估 batch_size
temperature:    # 0-110°C, step=2, noise/block_mse 温度
nonideal:       # original/improved 非理想参数 + 分档定义
outputs:        # 输出根目录
```

### 调参重点

修改 `nonideal` 组来控制退化/补偿效果：

| 参数 | 作用 | 调大 | 调小 |
|---|---|---|---|
| `ka` | 增益温度漂移系数 | 高温退化加快 | 高温退化减慢 |
| `kb` | 偏移温度漂移系数 | 同上 | 同上 |
| `sigma0` | 基准噪声强度 | 全温区退化增大 | 全温区退化减小 |
| `lam` | 高温噪声增长系数 | 高温噪声更大 | 高温噪声更小 |
| `nominal_gain` | 标称增益衰减 | 低温退化减小 | **低温退化增大** |
| `rho` | 补偿噪声缩放 | **补偿减弱** | 补偿增强 |
| `compensation_strength` | 补偿强度 | 补偿更接近档位中心 | 补偿减弱 |

## 5. 可用模型

| 模型名 | 架构 | 残差块数 | 块类型 |
|---|---|---|---|
| `resnet18_cifar` | ResNet18 | 8 | BasicBlock |
| `resnet34_cifar` | ResNet34 | 16 | BasicBlock |
| `resnet50_cifar` | ResNet50 | 16 | Bottleneck |

添加新模型：创建模型文件 → `@register_model` + `@register_injection` 装饰器 → 在 `experiment.yaml` 的 `models.active` 中添加。

## 6. 测试

```bash
pytest -v
```

26 个测试覆盖：配置加载、模型工厂、注入点注册、温度模型、hook 管理、指标计算。

## 7. 常见问题

### 服务器 GPU 被 `gpu_partition` 拦截

```bash
# 方案一：只用 CPU
CUDA_VISIBLE_DEVICES="" python run_experiment.py --models resnet18_cifar

# 方案二：绕过（需确认不与其他用户冲突）
LD_PRELOAD="" python run_experiment.py --models resnet18_cifar
```

### Improved 补偿过强（太接近 Ideal）

优先调整：
- 提高 `rho`（如 0.75 → 0.85）
- 提高 `compensation_strength`（如 0.55 → 0.65）

### Original 高温下降太快

优先降低：
- `ka`（如 0.00040 → 0.00030）
- `lam`（如 0.006 → 0.004）

### Original 退化不足

优先降低：
- `nominal_gain`（如 0.960 → 0.940）
- 提高 `sigma0`（如 0.040 → 0.050）
