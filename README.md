# MiniGPT

[![CI](https://github.com/qhr789/MiniGPT/actions/workflows/ci.yml/badge.svg)](https://github.com/qhr789/MiniGPT/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.2-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个从零实现的字符级、decoder-only MiniGPT，用于理解 Transformer、自回归语言模型训练和文本生成。

> 这是一个强调可读性与实验过程的学习基线，不是生产级大语言模型实现。

## 项目特点

- 字符级 Tokenizer 与训练集/验证集切分
- Token Embedding + 可学习位置嵌入
- 多头因果自注意力（Multi-Head Causal Self-Attention）
- Pre-LayerNorm、残差连接、Dropout 和 FFN
- 交叉熵训练、Loss / Perplexity 评估
- Best / Final Checkpoint 与 Early Stopping
- Greedy、Temperature、Top-k、Top-p 生成策略
- 不依赖外部模型权重，代码结构适合逐段阅读

## 基线结果

以下数据来自仓库内现有的一次训练记录：

| Epoch | Train Loss | Val Loss | 现象 |
|---:|---:|---:|---|
| 1 | 1.452 | 1.519 | 当前最佳验证集结果 |
| 10 | 1.093 | 1.653 | 训练集继续下降 |
| 20 | 1.060 | 1.676 | 验证集变差，出现过拟合 |

当前最佳模型为 `checkpoints/minigpt_best.pth`。由于数据量和模型规模都很小，生成质量有限，更适合用于观察训练、评估和解码策略，而不是追求文本质量。

## 快速开始

### 1. 环境要求

- Python 3.10+
- PyTorch 2.2+
- 可选：Apple Silicon MPS 或 NVIDIA CUDA

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需运行测试和代码检查：

```bash
python -m pip install -r requirements-dev.txt
```

### 2. 快速检查

```bash
python scripts/sanity_check.py
```

该命令会创建一个小型随机模型，检查前向传播、输出形状和 Loss 是否正常。

### 3. 生成文本

默认加载 `checkpoints/minigpt_best.pth`：

```bash
python generate.py \
  --prompt "First Citizen:" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9
```

使用 Greedy 解码：

```bash
python generate.py --prompt "First Citizen:" --greedy
```

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/minigpt_best.pth --split val
```

只评估一个 batch，便于快速检查：

```bash
python evaluate.py \
  --checkpoint checkpoints/minigpt_best.pth \
  --split val \
  --max-batches 1
```

### 5. 重新训练

```bash
python train.py
```

训练脚本会自动选择 MPS、CUDA 或 CPU，并生成：

```text
checkpoints/
├── minigpt_best.pth   # 验证集 Loss 最低的 checkpoint
├── minigpt_final.pth  # 最后一个 epoch 的 checkpoint
└── history.json       # 每个 epoch 的训练记录
```

## 项目结构

```text
.
├── .github/workflows/ci.yml    # GitHub Actions 配置
├── checkpoints/                # 已有模型权重与训练历史
├── data/input.txt              # Shakespeare 字符级数据
├── docs/LEARNING_GUIDE.md      # 分阶段学习与实验指南
├── minigpt/
│   ├── __init__.py             # 包导出
│   ├── checkpoint.py           # Checkpoint 加载
│   └── model.py                # Transformer 模型定义
├── scripts/sanity_check.py     # 前向传播与 Loss 快速检查
├── tests/test_model.py         # 模型形状与因果掩码测试
├── evaluate.py                 # Loss / Perplexity 评估入口
├── generate.py                 # 文本生成入口
├── train.py                    # 训练入口
├── EXPERIMENT_LOG.md           # 实验记录模板
├── pyproject.toml              # Pytest / Ruff 配置
├── requirements.txt
└── requirements-dev.txt
```

## 核心实现

模型入口位于 `minigpt/model.py`，包含：

- `Embedding`：字符 Token 与位置嵌入
- `SelfAttention`：多头因果自注意力
- `FeedForward`：Transformer Block 内的位置前馈网络
- `TransformerBlock`：Pre-LayerNorm 与双残差连接
- `MiniGPT`：完整 decoder-only 语言模型

默认训练配置集中放在 `train.py` 顶部，包括上下文长度、模型宽度、层数、注意力头数和训练轮数。

## 测试与代码检查

```bash
python -m pytest
python -m ruff check .
```

CI 会在 push 和 pull request 时自动执行代码检查、单元测试和 sanity check。

## 学习指南

- 想理解生成策略、Perplexity、Attention 可视化和受控实验：阅读 [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md)
- 想记录实验结果：使用 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)

## 已知限制

- 只支持字符级建模，没有使用 BPE 或 SentencePiece。
- 训练数据仅约 1 MB，模型和上下文长度都很小。
- 当前结果来自有限次数的单次训练，不代表稳定复现结论。
- 尚未提供 Attention 权重可视化。
- 模型不能用于生产推理，也不适合作为通用文本生成服务。

## 贡献

欢迎提交问题、测试和可复现实验。提交前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

本项目使用 [MIT License](LICENSE)。
