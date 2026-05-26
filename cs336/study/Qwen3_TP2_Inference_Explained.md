# vLLM Qwen3 推理过程解析 (TP=2)

本文档解释当你使用 `tensor_parallel_size=2` 运行 Qwen3 模型时，vLLM 内部发生的关键过程。

## 📋 目录

1. [张量并行 (Tensor Parallelism) 原理](#1-张量并行原理)
2. [模型加载与权重分片](#2-模型加载与权重分片)
3. [推理执行流程](#3-推理执行流程)
4. [关键日志解读](#4-关键日志解读)
5. [调试脚本使用说明](#5-调试脚本使用说明)

---

## 1. 张量并行原理

### 1.1 什么是张量并行？

张量并行 (Tensor Parallelism, TP) 是将模型的每一层**按维度切分**到多个 GPU 上。与数据并行不同，TP 不是复制整个模型，而是让每个 GPU 只持有模型参数的一部分。

### 1.2 Qwen3 8B 在 TP=2 下的分片方式

```
原始模型 (单 GPU):
┌─────────────────────────────────────┐
│ hidden_size = 4096                  │
│ num_attention_heads = 32            │
│ num_kv_heads = 8 (GQA)              │
│ intermediate_size = 14336           │
└─────────────────────────────────────┘

TP=2 分片后:
┌─────────────────────┐  ┌─────────────────────┐
│     GPU 0           │  │     GPU 1           │
├─────────────────────┤  ├─────────────────────┤
│ num_heads = 16      │  │ num_heads = 16      │
│ num_kv_heads = 4    │  │ num_kv_heads = 4    │
│ q_size = 2048       │  │ q_size = 2048       │
│ kv_size = 512       │  │ kv_size = 512       │
└─────────────────────┘  └─────────────────────┘
```

### 1.3 两种关键的并行线性层

#### ColumnParallelLinear (列并行)
```
输入: [batch, seq_len, hidden_size]
权重: [hidden_size, output_size/TP]  # 输出维度被切分
输出: [batch, seq_len, output_size/TP]  # 每个 GPU 有部分输出

应用于: QKV 投影 (qkv_proj), FFN 的 gate_up_proj
```

#### RowParallelLinear (行并行)
```
输入: [batch, seq_len, input_size/TP]  # 输入已经被切分
权重: [input_size/TP, output_size]
输出: [batch, seq_len, output_size]  # 需要 AllReduce 求和

应用于: 输出投影 (o_proj), FFN 的 down_proj
```

---

## 2. 模型加载与权重分片

### 2.1 加载流程

```
1. 解析模型配置 (Qwen3Config)
   ↓
2. 创建模型骨架 (Qwen3ForCausalLM)
   - 每个 GPU 创建 **分片后** 的层
   ↓
3. 加载权重
   - vLLM 自动将 HuggingFace 权重切分到对应 GPU
   ↓
4. 初始化 KV Cache
   - 根据可用显存分配 PagedAttention blocks
```

### 2.2 QKVParallelLinear 权重分片

```python
# 原始 Qwen3 8B 配置
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8  # GQA: Grouped Query Attention
head_dim = 4096 / 32 = 128

# TP=2 分片后 (每个 GPU)
num_heads = 32 / 2 = 16
num_kv_heads = 8 / 2 = 4

# 每个 GPU 的 QKV 权重形状
q_proj: [4096, 2048]   # 4096 -> 16 heads * 128
k_proj: [4096, 512]    # 4096 -> 4 kv_heads * 128
v_proj: [4096, 512]    # 4096 -> 4 kv_heads * 128

# 合并后的 qkv_proj 权重
qkv_proj: [4096, 2048+512+512] = [4096, 3072]
```

### 2.3 日志中观察权重分片

运行调试脚本后，你会看到类似这样的日志：

```
[TP_RANK=0] 🔧 Qwen3Attention 初始化:
    ├── hidden_size: 4096
    ├── total_num_heads: 32 (全部注意力头数)
    ├── num_heads (本分片): 16 (= 32 / 2)
    ├── total_num_kv_heads: 8 (全部KV头数)
    ├── num_kv_heads (本分片): 4
    ├── head_dim: 128
    ├── q_size (本分片): 2048 (= 16 * 128)
    └── kv_size (本分片): 512 (= 4 * 128)
```

---

## 3. 推理执行流程

### 3.1 单个 Decoder Layer 的执行流程

```
输入: hidden_states [batch, seq, 4096]
     ↓
┌────────────────────────────────────────────────────┐
│  1. RMSNorm (input_layernorm)                      │
│     - 每个 GPU 独立计算，无通信                      │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│  2. Self-Attention                                  │
│     ┌──────────────────────────────────────────┐   │
│     │ 2.1 QKV Projection (ColumnParallelLinear) │   │
│     │     - 每个 GPU 计算部分 Q, K, V             │   │
│     │     - GPU0: Q[:16], K[:4], V[:4]          │   │
│     │     - GPU1: Q[16:], K[4:], V[4:]          │   │
│     └──────────────────────────────────────────┘   │
│                      ↓                              │
│     ┌──────────────────────────────────────────┐   │
│     │ 2.2 QK-Norm (RMSNorm on Q and K)          │   │
│     │     - Qwen3 特有，每个 GPU 独立计算          │   │
│     └──────────────────────────────────────────┘   │
│                      ↓                              │
│     ┌──────────────────────────────────────────┐   │
│     │ 2.3 RoPE (旋转位置编码)                     │   │
│     │     - 每个 GPU 独立应用到 Q, K              │   │
│     └──────────────────────────────────────────┘   │
│                      ↓                              │
│     ┌──────────────────────────────────────────┐   │
│     │ 2.4 Attention (PagedAttention/FlashAttn)  │   │
│     │     - 每个 GPU 计算自己负责的 heads         │   │
│     │     - 使用 KV Cache                        │   │
│     └──────────────────────────────────────────┘   │
│                      ↓                              │
│     ┌──────────────────────────────────────────┐   │
│     │ 2.5 Output Projection (RowParallelLinear) │   │
│     │     - 每个 GPU 计算部分结果                  │   │
│     │     - ⚡ AllReduce: 求和所有 GPU 的结果     │   │
│     └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│  3. Residual Connection                            │
│     hidden_states = hidden_states + attn_output    │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│  4. RMSNorm (post_attention_layernorm)             │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│  5. FFN (SwiGLU)                                   │
│     ┌──────────────────────────────────────────┐   │
│     │ 5.1 gate_up_proj (ColumnParallelLinear)   │   │
│     │     - 每个 GPU: [4096, 14336/2]           │   │
│     └──────────────────────────────────────────┘   │
│                      ↓                              │
│     ┌──────────────────────────────────────────┐   │
│     │ 5.2 SiLU + element-wise multiply          │   │
│     │     - 每个 GPU 独立计算                     │   │
│     └──────────────────────────────────────────┘   │
│                      ↓                              │
│     ┌──────────────────────────────────────────┐   │
│     │ 5.3 down_proj (RowParallelLinear)         │   │
│     │     - ⚡ AllReduce: 求和所有 GPU 的结果     │   │
│     └──────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────┐
│  6. Residual Connection                            │
│     output = hidden_states + ffn_output            │
└────────────────────────────────────────────────────┘
     ↓
输出: hidden_states [batch, seq, 4096]
```

### 3.2 通信模式

**每个 Decoder Layer 有 2 次 AllReduce:**
1. `o_proj` (Attention 输出投影)
2. `down_proj` (FFN 下投影)

**对于 Qwen3 8B (36 层):**
- 单次前向传播: 36 × 2 = **72 次 AllReduce**

---

## 4. 关键日志解读

### 4.1 模型初始化日志

```log
[TP_RANK=0] 🔧 QKVParallelLinear 初始化:
    ├── hidden_size: 4096
    ├── head_size: 128
    ├── total_num_heads: 32
    ├── num_heads (本分片): 16        # ← 32/2 = 16
    ├── total_num_kv_heads: 8
    ├── num_kv_heads (本分片): 4       # ← 8/2 = 4
    ├── num_kv_head_replicas: 1        # ← 不需要复制 KV heads
    └── output_sizes: [2048, 512, 512] # ← Q, K, V 各自的大小
```

### 4.2 Attention Forward 日志

```log
[TP_RANK=0] 🔄 Qwen3Attention.forward:
    ├── input hidden_states: torch.Size([1, 10, 4096]) torch.bfloat16
    ├── positions: torch.Size([10])
    
[TP_RANK=0] ✅ Qwen3Attention.forward 完成:
    └── output: torch.Size([1, 10, 4096])  # 输出维度不变（经过 AllReduce）
```

### 4.3 AllReduce 日志

```log
📡 AllReduce #1:
    ├── input shape: torch.Size([1, 10, 4096])
    ├── world_size: 2
    ├── rank: 0/2
    └── group: tp:0

📡 AllReduce #2:
    ├── input shape: torch.Size([1, 10, 4096])
    ...
```

---

## 5. 调试脚本使用说明

### 5.1 运行脚本

```bash
# 进入 WSL/Linux 环境
wsl

# 激活 vLLM 环境
conda activate vllm

# 运行调试脚本
cd /mnt/d/projects/vllm
python study/qwen3_tp2_inference_debug.py
```

### 5.2 修改模型

如果显存不足，可以修改脚本中的模型：

```python
# 小模型（测试用）
MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# 大模型（需要更多显存）
MODEL_NAME = "Qwen/Qwen3-8B"
```

### 5.3 查看完整日志

日志会输出到两个地方：
1. **控制台** - 实时查看
2. **文件** - `study/qwen3_tp2_debug.log`

### 5.4 自定义 Hook

你可以在 `inject_logging_hooks()` 函数中添加更多 hook，例如：

```python
# 观察 Attention 内部的 softmax
from vllm.attention import Attention
original_attn_forward = Attention.forward

def patched_attn_forward(self, q, k, v):
    study_logger.debug(f"Attention: Q={q.shape}, K={k.shape}, V={v.shape}")
    return original_attn_forward(self, q, k, v)

Attention.forward = patched_attn_forward
```

---

## 6. 与 CS336 的关联

这个调试脚本展示的概念与 CS336 Assignment 2 (Systems) 紧密相关：

| 概念 | CS336 内容 | vLLM 实现 |
|------|-----------|-----------|
| 张量并行 | Lecture 08 分布式训练 | `ColumnParallelLinear`, `RowParallelLinear` |
| FlashAttention | Assignment 2 核心作业 | `vllm.attention` backends |
| AllReduce | DDP 通信模式 | `GroupCoordinator.all_reduce` |
| KV Cache | Lecture 10 推理优化 | `PagedAttention` |

建议学习路径：
1. 先完成 CS336 Assignment 2 的 FlashAttention 实现
2. 运行本调试脚本观察真实的分布式推理
3. 阅读 vLLM 源码中对应的实现

---

## 附录：关键源码位置

| 功能 | 文件路径 |
|------|----------|
| Qwen3 模型定义 | `vllm/model_executor/models/qwen3.py` |
| 并行线性层 | `vllm/model_executor/layers/linear.py` |
| 分布式状态管理 | `vllm/distributed/parallel_state.py` |
| 模型执行器 | `vllm/v1/worker/gpu_model_runner.py` |
| Attention 实现 | `vllm/attention/` |
