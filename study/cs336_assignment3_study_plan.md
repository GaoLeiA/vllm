# CS336 Assignment 3: Scaling - 学习计划

> **目标**: 理解 Scaling Laws —— 如何在训练前预测模型性能，以及如何最优分配计算预算

---

## 📋 Assignment 3 任务概览

Assignment 3 与前面的编程任务不同，它更偏**理论分析和实验**:

| 模块 | 任务描述 | 难度 | vLLM 对应概念 |
|------|---------|------|--------------|
| **Scaling Laws 推导** | 拟合 Chinchilla-style 的 scaling law 参数 | ⭐⭐⭐ | 选择何种大小的模型部署 |
| **IsoFLOP 曲线分析** | 分析固定 FLOP 下的最优模型/数据配置 | ⭐⭐⭐ | 理解算力-性能关系 |
| **FLOPs 计算** | 计算 Transformer 训练/推理的 FLOPs | ⭐⭐ | 推理效率评估 |
| **最优分配** | 给定计算预算，确定最优模型大小和数据量 | ⭐⭐⭐ | 实际工程决策 |

---

## 🎯 核心知识点

### 1. Scaling Laws 基础

**核心公式** (Chinchilla-style):

```
L(N, D) = A/N^α + B/D^β + E

其中:
  L = 测试损失 (越小越好)
  N = 模型参数量 (non-embedding)
  D = 训练数据量 (token 数)
  A, B, E, α, β = 需要拟合的常数
```

**关键洞见**:
- 损失可以分解为三个来源: 模型能力不足 (A/N^α)、数据不足 (B/D^β)、不可约误差 (E)
- **Chinchilla 定律**: 模型参数和训练数据应该等比例增长
- 对 70B 参数模型，最优训练需要 ~1.4T tokens

---

### 2. IsoFLOP 分析

**IsoFLOP 曲线**: 固定总计算量 (FLOPs) 下，不同 (N, D) 组合的损失

```
     Loss
      │
      │    C₁ (小 FLOP)
      │   ╱  ╲
      │  ╱    ╲
      │ ╱      ╲     C₂ (中 FLOP)
      │╱   *    ╲   ╱  ╲
      │    最优   ╲ ╱    ╲
      │          ╲╱  *   ╲    C₃ (大 FLOP)
      │           │ 最优  ╲  ╱  ╲
      │           │        ╲╱ *  ╲
      └───────────┼─────────┼─────→ 模型参数量 N
                  N₁*       N₂*    N₃*
```

**数据文件**: `data/isoflops_curves.json` 包含了训练实验数据

---

### 3. FLOPs 计算

```
Transformer FLOPs 计算 (近似):

训练 FLOPs ≈ 6 × N × D
  N = 非嵌入参数量
  D = 训练 token 数
  6 = 2 (前向) × 3 (前向+反向)

推理 FLOPs ≈ 2 × N × T
  T = 生成的 token 数

更精确的计算 (per layer):
  MLP:     6 × B × T × D × F   (3 个矩阵: Wup, Wgate, Wdown)
  Attn:    4 × B × T × D × D   (QKV projection + output)
  + Attn:  4 × B × S × T × D   (attention computation)
```

---

### 4. 最优计算分配

给定总计算预算 C (in FLOPs):

```python
# Chinchilla 最优分配
# C ≈ 6 × N × D
# 需要最小化 L(N, D) subject to C = 6·N·D

# 结论: N_opt ∝ C^a,  D_opt ∝ C^b,  其中 a + b = 1
# Chinchilla: a ≈ 0.50, b ≈ 0.50 (等比例缩放)
# Kaplan (OpenAI): a ≈ 0.73, b ≈ 0.27 (更偏向增大模型)
```

---

## 🎯 学习路径

### Phase 1: 理解 Scaling Laws 理论 (3小时)

**必读论文**:
1. [Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556) — 最重要
2. [Kaplan et al., 2020](https://arxiv.org/abs/2001.08361) — OpenAI 早期工作
3. [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — 数据约束

**核心问题**:
- 给定 X FLOPs 的预算，应该训练多大的模型？用多少数据？
- 如果数据量有限（不能增加更多数据），该怎么办？
- Scaling law 的预测在多大范围内可靠？

---

### Phase 2: FLOPs 计算实践 (2小时)

用 `model.py` 中的 `BasicsTransformerLM` 进行参数量和 FLOPs 计算:

```python
# 参考模型配置
config = {
    'vocab_size': 32000,
    'context_length': 2048,
    'd_model': 4096,
    'num_layers': 32,
    'num_heads': 32,
    'd_ff': 11008,
}

# 参数量 (non-embedding)
# 每层: 4*d*d (attention) + 3*d*d_ff (SwiGLU) + 2*d (RMSNorm)
# 总计: num_layers * (4*d² + 3*d*d_ff + 2*d) + V*d + d (lm_head + final_ln)
```

---

### Phase 3: IsoFLOP 曲线拟合 (3小时)

使用 `data/isoflops_curves.json` 中的数据:

```python
import json
import numpy as np
from scipy.optimize import curve_fit

# 加载数据
with open('data/isoflops_curves.json') as f:
    data = json.load(f)

# Scaling law 模型
def scaling_law(params, A, alpha, B, beta, E):
    N, D = params
    return A / N**alpha + B / D**beta + E

# 拟合参数
popt, pcov = curve_fit(scaling_law, ...)

# 绘制 IsoFLOP 曲线
# 对每条曲线，找到最小 loss 对应的 N_opt
```

---

### Phase 4: 延伸分析 (2小时)

- **超越 Chinchilla**: 实际中很多模型是 "over-trained" 的 (如 Llama 3)
- **推理时计算**: 推理时的 FLOPs 只依赖 N，所以更小的模型推理更便宜
- **数据复用**: 当优质数据有限时，重复数据有多大影响？

---

## 🔗 与 vLLM 的连接

Scaling Laws 直接影响 **vLLM 需要服务的模型规模**:

| Scaling 概念 | vLLM 实际影响 |
|-------------|-------------|
| 模型参数量 N | 决定 GPU 显存需求、是否需要 Tensor Parallelism |
| 训练数据量 D | 决定模型质量 |
| 推理 FLOPs | 决定延迟和吞吐量 |
| KV Cache 大小 | 与 N·L·K·H 成正比，Scaling 影响模型架构选择 |

**实际案例分析**:
```
Llama 2 系列:
  7B  → 单 GPU 推理，低延迟
  13B → 单 GPU 勉强，可能需要量化
  70B → 需要多 GPU，Tensor Parallelism
  
Chinchilla 最优 vs 实际:
  Llama 3 8B 训练了 15T tokens (远超 Chinchilla 最优的 ~160B)
  原因: 推理时 FLOPs 低 → 用更多训练时间换更便宜的推理
```

---

## 📚 配套讲座

1. **Lecture 9** (`nonexecutable/2025 Lecture 9 - Scaling laws basics.pdf`) — Scaling Laws 基础理论
2. **Lecture 10** (`spring2025-lectures/lecture_10.py`) — 推理中的 FLOPs 和延迟分析
3. **Lecture 11** (`nonexecutable/2025 Lecture 11 - Scaling details.pdf`) — Scaling 进阶

---

## 📅 学习时间安排

| 阶段 | 预计时间 | 产出 |
|------|---------|------|
| Phase 1: Scaling Laws 理论 | 3小时 | 理解 Chinchilla 论文 |
| Phase 2: FLOPs 计算 | 2小时 | 能够手算 Transformer FLOPs |
| Phase 3: IsoFLOP 拟合 | 3小时 | 拟合参数、绘制曲线 |
| Phase 4: 延伸分析 | 2小时 | 理解实际工程权衡 |

**总计**: ~10小时

---

## ✅ 关键学习成果

完成 Assignment 3 后，您将理解：
- 📈 **为什么 Scaling Laws 重要**: 可以在训练前预测性能
- 🧮 **如何计算 FLOPs**: 给定模型配置，精确估算计算量
- ⚖️ **最优计算分配**: 给定预算如何选择模型大小和数据量
- 💰 **实际工程权衡**: 训练成本 vs 推理成本的权衡 (over-training)
