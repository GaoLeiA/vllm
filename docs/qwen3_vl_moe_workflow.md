# Qwen3-VL-30B-A3B-Instruct 推理流程详解

本文档详细讲解 Qwen3-VL-30B-A3B-Instruct 模型的完整推理流程。

## 1. 模型架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen3-VL-30B-A3B-Instruct                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌──────────────────────────────────┐  │
│  │  Vision Encoder │     │      Language Model (MoE)        │  │
│  │  ───────────────│     │  ────────────────────────────────│  │
│  │  • Patch Embed  │────▶│  • Token Embedding               │  │
│  │  • ViT Blocks   │     │  • Decoder Layers ×N             │  │
│  │  • Patch Merger │     │    - Self-Attention (mRoPE)      │  │
│  └─────────────────┘     │    - MoE FFN (128 experts, top8) │  │
│                          │  • LM Head                       │  │
│                          └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**关键参数:**
| 参数 | 值 |
|------|-----|
| 总参数量 | ~30B |
| 激活参数 | ~3B (每 token) |
| 专家数量 | 128 routed experts |
| Top-K | 8 |

---

## 2. 推理流程三阶段

### 2.1 Vision Encoding (视觉编码)

```
输入图像 [448×448×3]
    │
    ▼
Patch Embedding (3D Conv, 14×14 patches)
    │ → [1024 patches, 1280-dim]
    ▼
ViT Transformer Blocks (×depth)
    │ + 2D RoPE 位置编码
    ▼
Patch Merger (2×2 空间下采样)
    │ → [256 patches, 3584-dim]
    ▼
输出: Vision Embeddings
```

### 2.2 Prefill (预填充)

```
文本 Tokens + Vision Embeddings
    │
    ▼
Token Embedding Merge
    │ → [seq_len, hidden_size]
    ▼
Decoder Layers ×N
    ├── Self-Attention (完整 causal mask)
    └── MoE FFN (每 token 选 top-8 专家)
    │
    ▼
生成 KV Cache + 第一个输出 Token
```

### 2.3 Decode (解码)

```
新 Token (1个)
    │
    ▼
Token Embedding
    │ → [1, hidden_size]
    ▼
Decoder Layers ×N
    ├── Self-Attention (使用 KV Cache)
    └── MoE FFN
    │
    ▼
预测下一个 Token
    │
    ▼
循环直到 EOS
```

---

## 3. MoE (Mixture of Experts) 层详解

### 3.1 路由机制

```
hidden_states [num_tokens, hidden_size]
    │
    ▼
Router (Linear Layer)
    │ → router_logits [num_tokens, 128]
    ▼
Top-K Selection (K=8)
    │ → topk_weights [num_tokens, 8]
    │ → topk_ids [num_tokens, 8]
    ▼
Expert FFN Computation (并行)
    │
    ▼
Weighted Aggregation
    │ → output = Σ(weight_i × expert_i(x))
    ▼
output [num_tokens, hidden_size]
```

### 3.2 Expert FFN 结构

每个专家是一个标准的 SwiGLU FFN:
```python
gate = W_gate @ x          # 门控投影
up = W_up @ x              # 上投影
hidden = SiLU(gate) * up   # 激活
output = W_down @ hidden   # 下投影
```

### 3.3 计算效率

- **Dense FFN**: `O(tokens × hidden × intermediate)`
- **Sparse MoE**: `O(tokens × hidden × intermediate × 8/128)`
- **节省比例**: 93.75% 参数不参与计算

---

## 4. Prefill vs Decode 对比

| 特性 | Prefill | Decode |
|------|---------|--------|
| 输入长度 | 完整 prompt + 图像 | 1 个 token |
| 计算特点 | Compute-bound | Memory-bound |
| Attention | Full causal (N×N) | Query=1, KV=历史 |
| KV Cache | 生成并保存 | 读取并更新 |
| Vision Encoder | 执行 | 跳过 |
| 执行次数 | 1 次 | 每生成 token 1 次 |

---

## 5. 源码位置

| 组件 | 文件路径 |
|------|---------|
| 主模型 | `vllm/model_executor/models/qwen3_vl_moe.py` |
| Vision Encoder | `vllm/model_executor/models/qwen3_vl.py` |
| MoE Block | `vllm/model_executor/models/qwen3_moe.py` |
| FusedMoE Layer | `vllm/model_executor/layers/fused_moe/layer.py` |
| MoE Kernels | `vllm/model_executor/layers/fused_moe/fused_moe.py` |
