# MiniGPT：从零实现的小型 GPT

这是一个用于理解 Transformer、语言模型训练和文本生成的字符级 MiniGPT。
它的定位是**基础学习项目**，不是简历主线项目；目标是在几天内收尾，把关键机制真正讲清楚。

## 当前状态

已经完成：

- 字符级 Tokenizer 和数据切分
- Token Embedding + Position Embedding
- Multi-Head Self-Attention
- Causal Mask
- FeedForward
- LayerNorm、Residual、Dropout
- Transformer Block
- LM Head 和交叉熵训练
- 训练集/验证集 Loss
- Best/Final Checkpoint
- 基础采样生成
- Loss、Perplexity 评估脚本

已有训练日志显示：

| Epoch | Train Loss | Val Loss | 现象 |
|---:|---:|---:|---|
| 1 | 1.452 | 1.519 | 当前最佳 |
| 10 | 1.093 | 1.653 | 训练集继续下降 |
| 20 | 1.060 | 1.676 | 验证集变差 |

结论：项目已经跑通，但模型明显过拟合。现阶段不要盲目继续堆 Epoch，应该先用最佳模型完成生成策略、评估、可视化和受控实验。

## 项目结构

```text
MiniGPT/
├── data/
│   └── input.txt              # Shakespeare 字符级数据
├── model/
│   ├── __init__.py
│   └── minigpt.py             # 只保留模型定义
├── scripts/
│   └── sanity_check.py        # 模型前向和 Loss 快速检查
├── train/
│   ├── train.py               # 训练入口
│   ├── minigpt_best.pth       # 最佳验证集模型
│   └── minigpt_final.pth      # 最后一轮模型
├── utils/
│   └── checkpoint.py          # Checkpoint 加载
├── evaluate.py                # Loss / Perplexity
├── generate.py                # Greedy / Temperature / Top-k / Top-p
├── EXPERIMENT_LOG.md          # 你要亲手填写的实验记录
├── requirements.txt
└── README.md
```

## 运行方式

所有命令都在 `MiniGPT` 根目录执行。

先做一次快速检查：

```bash
python scripts/sanity_check.py
```

如需重新训练：

```bash
python train/train.py
```

训练脚本会：

- 自动使用 MPS、CUDA 或 CPU
- 固定随机种子
- 保存最佳模型和最终模型
- 保存 `train/history.json`
- 使用 early stopping，避免验证集持续变差时浪费时间

生成文本时默认加载 `minigpt_best.pth`：

```bash
python generate.py \
  --prompt "First Citizen:" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9
```

Greedy：

```bash
python generate.py --prompt "First Citizen:" --greedy
```

评估最佳模型：

```bash
python evaluate.py --checkpoint train/minigpt_best.pth --split val
```

快速评估一个 batch：

```bash
python evaluate.py --checkpoint train/minigpt_best.pth --split val --max-batches 1
```

## 你现在还要做什么

下面这些不是“知道有这个东西”就够了，必须自己运行、观察、解释并记录。代码脚手架已经整理好，但结论要由你形成。

### 1. 生成策略：必须亲手理解

阅读 `generate.py` 中的 `sample_next_token()`，依次运行：

```bash
python generate.py --prompt "First Citizen:" --greedy
python generate.py --prompt "First Citizen:" --temperature 0.5
python generate.py --prompt "First Citizen:" --temperature 1.0
python generate.py --prompt "First Citizen:" --temperature 1.5
python generate.py --prompt "First Citizen:" --top-k 10
python generate.py --prompt "First Citizen:" --top-p 0.9
```

你要能回答：

- Greedy 为什么容易重复或陷入固定表达？
- Temperature 增大或减小后，概率分布发生了什么？
- Top-k 和 Top-p 分别在“截掉什么”？
- 为什么通常先除以 Temperature，再过滤，再 Softmax？
- 为什么同样的 prompt 每次输出会不同？

把结果填写到 `EXPERIMENT_LOG.md` 的“实验 1”。

### 2. 理解 Perplexity 和过拟合

分别运行：

```bash
python evaluate.py --checkpoint train/minigpt_best.pth --split train
python evaluate.py --checkpoint train/minigpt_best.pth --split val
python evaluate.py --checkpoint train/minigpt_final.pth --split train
python evaluate.py --checkpoint train/minigpt_final.pth --split val
```

你要能回答：

- Perplexity 与交叉熵 Loss 的关系是什么？
- 为什么最终模型训练 Loss 更低，验证 Loss 反而更高？
- 为什么不能只看训练集 Loss 判断模型是否变好？
- 为什么生成文本默认应该加载 `best` 而不是 `final`？

把结果填写到 `EXPERIMENT_LOG.md` 的“实验 2”。

### 3. Attention 可视化：这是当前最大的代码扩展点

目前 `SelfAttention.forward()` 只返回注意力输出，不返回 Attention Weights。
下一步需要你在理解 Q、K、V 的基础上增加一个可选返回路径：

```text
Q, K, V
  ↓
Q @ K^T / sqrt(head_dim)
  ↓
Causal Mask
  ↓
Softmax
  ↓
Attention Weights
```

目标是选定一句短文本，观察某个 token 实际关注了哪些前面的 token。
这一步不要只复制代码，先手算一个 3~4 token 的 toy example，再实现可视化。

### 4. 做受控实验

每次只改变一个变量：

- 模型规模：`(128, 4 layers, 4 heads)` 对比 `(256, 6 layers, 8 heads)`
- 上下文长度：`32`、`64`、`128`
- 生成策略：Greedy、Temperature、Top-k、Top-p

每个实验记录：

```text
参数量
训练时间
Best Val Loss
Val Perplexity
生成样本
你的解释
```

表格已经放在 `EXPERIMENT_LOG.md`。

### 5. 最后再考虑 LoRA

LoRA 不应该现在立刻开始。先完成上面的生成、评估、Attention 和规模实验。
之后可以冻结 MiniGPT 原参数，只给 `q_proj`、`v_proj` 增加低秩矩阵，比较：

```text
Full training vs LoRA
参数量
显存/内存
训练时间
Val Loss
生成效果
```

LoRA 在这里主要用于理解机制，不是这个学习项目必须追求的效果指标。

## MiniGPT 的完成标准

满足下面几条就可以收尾：

- 能不看代码解释 Attention 的每一步和张量形状
- 能解释 Causal Mask、LayerNorm、Residual、FFN 的作用
- 能比较 Greedy、Temperature、Top-k、Top-p
- 能解释 Perplexity 和过拟合
- 能展示至少一张 Attention 可视化图
- 至少完成一组模型规模或上下文长度对比实验
- 能说清项目局限：字符级、数据小、模型小、生成一致性有限

## 时间建议

- 生成策略：1 天
- Perplexity 和过拟合：半天
- Attention 可视化：1~2 天
- 受控实验：1 天
- LoRA：按时间选做

做完这些就收尾，进入上一版总路线中的更高优先级项目：垂直场景 RAG。不要把 MiniGPT 扩展成长期主线。
