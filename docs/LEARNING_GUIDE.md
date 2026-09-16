# MiniGPT 学习与实验指南

本指南面向希望逐项理解 Transformer、语言模型训练和文本生成的读者。所有命令都应在仓库根目录执行。

## 1. 生成策略

阅读 `generate.py` 中的 `sample_next_token()`，依次比较：

```bash
python generate.py --prompt "First Citizen:" --greedy
python generate.py --prompt "First Citizen:" --temperature 0.5
python generate.py --prompt "First Citizen:" --temperature 1.0
python generate.py --prompt "First Citizen:" --temperature 1.5
python generate.py --prompt "First Citizen:" --top-k 10
python generate.py --prompt "First Citizen:" --top-p 0.9
```

需要解释：

- Greedy 为什么容易重复或陷入固定表达？
- Temperature 增大或减小后，概率分布发生了什么？
- Top-k 和 Top-p 分别截掉了什么？
- 为什么通常先除以 Temperature，再过滤，最后 Softmax？
- 为什么相同 Prompt 在随机采样下可能得到不同输出？

将观察记录到 `EXPERIMENT_LOG.md` 的“实验 1”。

## 2. Perplexity 与过拟合

分别运行：

```bash
python evaluate.py --checkpoint checkpoints/minigpt_best.pth --split train
python evaluate.py --checkpoint checkpoints/minigpt_best.pth --split val
python evaluate.py --checkpoint checkpoints/minigpt_final.pth --split train
python evaluate.py --checkpoint checkpoints/minigpt_final.pth --split val
```

需要解释：

- Perplexity 与交叉熵 Loss 的关系是什么？
- 为什么 Final 模型训练 Loss 更低，验证 Loss 反而更高？
- 为什么不能只根据训练集 Loss 判断模型是否变好？
- 为什么生成文本默认加载 Best Checkpoint？

将观察记录到 `EXPERIMENT_LOG.md` 的“实验 2”。

## 3. Attention 可视化

目前 `SelfAttention.forward()` 只返回注意力输出，不返回 Attention Weights。下一步可以为它增加可选返回路径：

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

建议先手算一个 3 到 4 个 Token 的示例，再选择一段短文本，观察某个 Token 实际关注了哪些前面的 Token。

## 4. 受控实验

每次只改变一个变量：

- 模型规模：`(embedding_dim=128, num_layers=4, num_heads=4)` 对比 `(256, 6, 8)`
- 上下文长度：`32`、`64`、`128`
- 生成策略：Greedy、Temperature、Top-k、Top-p

每组实验至少记录：

```text
参数量
训练时间
Best Val Loss
Val Perplexity
生成样本
结论与解释
```

对应表格已经放在 `EXPERIMENT_LOG.md`。

## 5. LoRA 扩展

完成生成、评估、Attention 和规模实验后，可以冻结原始参数，只给 `q_proj`、`v_proj` 增加低秩矩阵，并比较：

```text
Full Training vs LoRA
参数量
显存 / 内存
训练时间
Val Loss
生成效果
```

LoRA 在这里主要用于理解参数高效微调，并不是必须达到的完成指标。

## 完成标准

- 能脱离代码解释 Attention 的每一步和张量形状
- 能解释 Causal Mask、LayerNorm、Residual 和 FFN
- 能比较 Greedy、Temperature、Top-k 和 Top-p
- 能解释 Perplexity 与过拟合
- 至少完成一张 Attention 可视化图
- 至少完成一组模型规模或上下文长度对比实验
- 能说明项目局限：字符级、数据小、模型小、生成一致性有限
