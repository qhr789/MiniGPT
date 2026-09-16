# MiniGPT 实验记录

每个实验只改变一个变量。运行前先写下假设，运行后填写结果，不要只贴截图。

## 实验 1：生成策略

| 编号 | 策略 | temperature | top-k | top-p | prompt | 观察 | 结论 |
|---|---|---:|---:|---:|---|---|---|
| G1 | Greedy | - | - | - | First Citizen: |  |  |
| G2 | Temperature | 0.5 | - | - | First Citizen: |  |  |
| G3 | Temperature | 1.0 | - | - | First Citizen: |  |  |
| G4 | Top-k | 1.0 | 10 | - | First Citizen: |  |  |
| G5 | Top-p | 1.0 | - | 0.9 | First Citizen: |  |  |

## 实验 2：最佳模型与最终模型

| 模型 | Epoch | Val Loss | Perplexity | 生成的典型问题 |
|---|---:|---:|---:|---|
| best |  |  |  |  |
| final |  |  |  |  |

## 实验 3：模型规模

| 参数 | embedding_dim | num_layers | num_heads | 参数量 | 训练时间 | Best Val Loss | Val PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 基线 | 128 | 4 | 4 |  |  |  |  |
| 更大 | 256 | 6 | 8 |  |  |  |  |

## 实验 4：上下文长度

| max_seq_len | Best Val Loss | Val PPL | 训练时间 | 生成表现 |
|---:|---:|---:|---:|---|
| 32 |  |  |  |  |
| 64 |  |  |  |  |
| 128 |  |  |  |  |

## 每次实验后必须回答

1. 我改了什么，其他条件是否完全不变？
2. 哪个指标变好了，哪个指标变差了？
3. 这是训练集还是验证集上的变化？
4. 结果是否符合预期，可能原因是什么？
5. 这个结论能否用于真实 LLM 项目，边界在哪里？
