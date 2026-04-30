# AI 管理指南

## 1. 这个文件的目的

这个文件给以后参与修改的 AI 或维护者使用。目标是让后来者快速读懂实验代码，知道先看哪些文件、哪些参数能动、哪些结论不能乱写，避免每次都从零重新分析。

## 2. 推荐阅读顺序

第一次接手项目时，按这个顺序读：

1. `说明.md`：先理解项目做什么，以及论文叙事边界。
2. `数据.md`：看当前结果、旧结果问题、论文可用结论。
3. `修改.md`：看历史调参过程和下一步建议。
4. `readme.md`：看如何运行、输出在哪里。
5. `run_experiment.py`：看当前一键实验的真实参数和流程。
6. `configs/base.yaml`、`configs/temp_original.yaml`、`configs/temp_improved_*.yaml`：看分步骤脚本参数。
7. `nonideal/temp_model.py`：看温度非理想和补偿公式。
8. `nonideal/hooks.py`：看非理想如何注入 residual block。
9. `engine/evaluate.py`：看温度扫描、噪声扫描、MSE、bin sweep 的评估逻辑。
10. `scripts/*.py`：看分步骤命令如何调用 engine。
11. `models/resnet_cifar.py`：看 CIFAR-10 ResNet18 和 residual block 暴露方式。
12. `tests/`：看现有测试如何保护行为。

不要先改模型结构。这个项目的主线是行为模型调参，不是重新设计网络。

## 3. 文件职责

| 文件或目录 | 职责 |
|---|---|
| `run_experiment.py` | 推荐主入口，集中参数、一键训练评估画图 |
| `run_all.py` | 旧的一键入口，按 `scripts/` 顺序运行 |
| `configs/` | 分步骤脚本使用的 YAML 配置 |
| `engine/train.py` | 训练循环 |
| `engine/evaluate.py` | 评估逻辑和 CSV 导出 |
| `engine/metrics.py` | accuracy、MSE、recovery ratio 等指标 |
| `engine/utils.py` | 数据加载、seed、checkpoint、目录工具 |
| `models/resnet_cifar.py` | CIFAR-10 版 ResNet18 |
| `nonideal/temp_model.py` | 温度非理想行为模型 |
| `nonideal/hooks.py` | 在 residual block 输出后注入非理想 |
| `scripts/plot_results.py` | 从 CSV 生成论文图 |
| `tests/` | 单元测试 |

## 4. 实验主线

实验流程是：

```text
训练 CIFAR-10 ResNet18
-> 冻结模型参数
-> residual block 输出后注册 hook
-> ideal/original/improved 三种模式评估
-> 导出 CSV
-> 画论文图
-> 根据曲线调参
```

非理想注入点是每个 residual block 的最终输出，不是在卷积内部，也不是只在 logits 上加噪声。

## 5. 调参优先级

### 5.1 控制 `original`

`original` 太稳定，高温下降不明显：

```text
增大 ka 或 kb
必要时增大 lam
```

`original` 崩得太快：

```text
降低 ka 或 kb
降低 lam
最后才考虑减少注入层数
```

`25°C` 附近精度太高或太低：

```text
调整 nominal_gain 或 sigma0
```

### 5.2 控制 `improved`

`improved` 太接近 `ideal`：

```text
提高 rho
降低 compensation_strength
减少分档数量
```

`improved` 提升不明显：

```text
降低 rho
提高 compensation_strength
适当增加分档数量
```

### 5.3 控制 bin sweep

如果 bin sweep 出现“分档越多反而越差”，先检查是否只在单温度点评估。单点评估容易被档位中心是否恰好对齐温度误导。推荐使用全温区平均。

## 6. 修改代码的规则

改代码前先确认要使用哪套入口：

- 推荐入口：`run_experiment.py`
- 分步骤入口：`scripts/*.py` + `configs/*.yaml`

如果改了 `run_experiment.py` 的参数，同时又打算用分步骤脚本运行，就必须同步修改 `configs/*.yaml`。

修改实验逻辑时优先保持这些约束：

- 不改变 CIFAR-10 数据集。
- 不改变 ResNet18 主体结构，除非明确要做结构实验。
- 不改变 `ideal` baseline 的定义。
- 不改变 accuracy、MSE、recovery ratio 的核心定义。
- `original` 与 `improved` 对比时使用相同随机 seed，保证噪声路径可比。
- 输出 CSV 字段名要稳定，因为画图脚本依赖这些列。

## 7. 结果验收标准

一轮结果是否适合论文，按下面检查：

| 检查项 | 合理现象 |
|---|---|
| `ideal` | 约 `93%`，随温度基本稳定 |
| `original` | 高温逐步下降，不突然掉到 `10%` |
| `improved` | 高于 `original`，但不长期贴近 `ideal` |
| 高温收益 | `T > 85°C` 后收益更明显 |
| 噪声扫描 | 噪声越大精度越低，`improved` 更平稳 |
| 逐块 MSE | 深层误差通常更大，补偿后 MSE 降低 |
| Bin sweep | 分档增加收益逐渐饱和 |

如果某组结果很好看但不符合物理直觉，优先相信物理直觉，不要为了图漂亮而硬写结论。

## 8. 论文写作边界

可以写：

- 温度相关非理想会导致推理精度退化。
- 温度自适应补偿能在高温区提供稳定恢复。
- 补偿能降低逐块特征误差，尤其对深层更明显。
- 分档数量增加有边际收益，但收益会饱和。

不要轻易写：

- 结果等同真实芯片测量。
- 补偿完全消除了硬件非理想。
- 全温区都提升 `5%~10%`，除非数据确实支持。
- 噪声鲁棒性显著增强，除非噪声扫描差距足够大。

## 9. 每次 AI 修改后的收尾动作

每次修改实验后，至少做这些事：

1. 说明改了哪些文件。
2. 说明为什么改。
3. 运行能承受成本范围内的测试或实验。
4. 把关键结果写入 `修改.md`。
5. 如果结果用于论文，把表格和结论同步到 `数据.md`。
6. 如果运行方式变了，更新 `readme.md`。
7. 如果代码阅读顺序或维护规则变了，更新本文件。
