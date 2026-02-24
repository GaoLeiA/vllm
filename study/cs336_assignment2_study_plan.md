# CS336 Assignment 2: Systems - 学习计划

> **目标**: 掌握 LLM 系统级优化技术，并将所学与 vLLM 源码实现对照理解

---

## 📋 Assignment 2 任务概览

Assignment 2 要求您实现以下核心内容：

| 模块 | 任务描述 | 难度 | vLLM 对应实现 |
|------|---------|------|--------------|
| **FlashAttention (PyTorch)** | 用纯 PyTorch 实现 FlashAttention forward/backward | ⭐⭐⭐ | `vllm/attention/backends/` |
| **FlashAttention (Triton)** | 用 Triton 编写 FlashAttention kernel | ⭐⭐⭐⭐ | `vllm/attention/ops/` |
| **DDP Individual Parameters** | 实现逐参数梯度同步的分布式数据并行 | ⭐⭐⭐ | `vllm/distributed/` |
| **DDP Bucketed** | 实现分桶梯度同步的 DDP (更高效) | ⭐⭐⭐⭐ | PyTorch DDP |
| **Sharded Optimizer** | 实现 ZeRO-style 优化器状态分片 | ⭐⭐⭐⭐ | DeepSpeed/FSDP |

---

## 🎯 学习路径

### Phase 1: 理解基础模型代码 (1小时)

**目标**: 熟悉 `cs336_basics` 中的参考实现

**文件位置**: `cs336\assignment2-systems\cs336-basics\cs336_basics\model.py`

**核心组件**:
```
BasicsTransformerLM
├── Embedding (词嵌入)
├── RotaryEmbedding (RoPE 位置编码)
├── TransformerBlock × num_layers
│   ├── RMSNorm
│   ├── CausalMultiHeadSelfAttention
│   │   ├── q_proj, k_proj, v_proj
│   │   ├── scaled_dot_product_attention  ← 需要优化的地方!
│   │   └── output_proj
│   └── SwiGLU (FFN)
└── lm_head
```

**✅ 练习**: 
1. 阅读 `scaled_dot_product_attention` 函数 (lines 400-432)
2. 理解朴素实现的内存复杂度: O(N²) 用于注意力矩阵

---

### Phase 2: FlashAttention 理论 (2小时)

**核心论文**: [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

**关键概念**:

#### 2.1 为什么需要 FlashAttention?

```
标准 Attention 问题:
┌─────────────────────────────────────────┐
│  Q @ K^T  →  需要具化 N×N 注意力矩阵     │
│  内存: O(N²)                            │
│  IO 复杂度: O(N² d)                     │
└─────────────────────────────────────────┘

FlashAttention 解决方案:
┌─────────────────────────────────────────┐
│  分块计算 (Tiling)                       │
│  利用 Online Softmax 算法               │
│  内存: O(N) — 不需要存储完整注意力矩阵   │
│  IO 复杂度: O(N² d² / M) 其中 M 是 SRAM │
└─────────────────────────────────────────┘
```

#### 2.2 Online Softmax 算法

这是 FlashAttention 的核心！

```python
# 传统 Softmax 需要两遍:
# 1. 计算 max(x) 用于数值稳定
# 2. 计算 exp(x - max) 和 sum

# Online Softmax 可以一遍完成:
def online_softmax_step(m_prev, l_prev, x_block):
    m_new = max(m_prev, max(x_block))    # 更新最大值
    l_new = l_prev * exp(m_prev - m_new) + sum(exp(x_block - m_new))  # 更新归一化因子
    return m_new, l_new
```

#### 2.3 FlashAttention 伪代码

```python
# Forward Pass (简化版)
def flash_attention_forward(Q, K, V, block_size):
    N = Q.shape[0]
    O = zeros(N, d)
    L = zeros(N)      # logsumexp
    M = full(N, -inf) # running max
    
    for i in range(0, N, block_size):
        Qi = Q[i:i+block_size]
        for j in range(0, N, block_size):
            Kj, Vj = K[j:j+block_size], V[j:j+block_size]
            
            # 在 SRAM 中计算注意力分数
            S_ij = Qi @ Kj.T / sqrt(d)
            
            # Online softmax 更新
            m_new = max(M[i:i+block_size], rowmax(S_ij))
            P_ij = exp(S_ij - m_new)
            l_new = exp(M[i:i+block_size] - m_new) * L[i:i+block_size] + rowsum(P_ij)
            
            # 增量更新输出
            O[i:i+block_size] = (exp(M - m_new) * O[i:i+block_size] + P_ij @ Vj) / l_new
            
            M[i:i+block_size] = m_new
            L[i:i+block_size] = l_new
    
    return O, L  # L 是 log(l)
```

**📚 推荐资源**:
- [FlashAttention 论文精读](https://www.youtube.com/watch?v=gMOAud7hZg4)
- [Triton FlashAttention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

---

### Phase 3: 实现 FlashAttention (3-4小时)

**任务 1**: PyTorch 版本 (纯 Python + PyTorch ops)

创建文件: `cs336\assignment2-systems\cs336_systems\flash_attention.py`

```python
import torch
import torch.autograd as autograd
from einops import einsum

class FlashAttentionPyTorch(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Args:
            Q: (batch, n_queries, d)
            K: (batch, n_keys, d)
            V: (batch, n_keys, d)
            is_causal: 是否使用因果掩码
        Returns:
            O: (batch, n_queries, d)
        """
        # TODO: 实现分块 attention
        # 1. 设置 block_size (e.g., 64)
        # 2. 初始化 O, L (logsumexp), M (running max)
        # 3. 双重循环遍历 Q 和 KV 的块
        # 4. 使用 online softmax 更新 O
        
        # 保存用于 backward
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        计算 dQ, dK, dV
        
        关键技巧: 重计算注意力分数而不是存储它们!
        """
        Q, K, V, O, L = ctx.saved_tensors
        # TODO: 实现 backward
        # 参考论文 Algorithm 2
        return dQ, dK, dV, None
```

**任务 2**: Triton 版本 (需要 GPU)

```python
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
    Q, K, V, O, L,  # 指针
    stride_qb, stride_qh, stride_qs, stride_qd,
    # ... 更多 strides
    N_QUERIES, N_KEYS, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # TODO: 实现 Triton kernel
    pass
```

---

### Phase 4: 分布式数据并行 DDP (3-4小时)

**任务**: 实现自己的 DDP wrapper

#### 4.1 DDP 基础原理

```
┌─────────────────────────────────────────────────────────┐
│                    DDP 训练流程                          │
├─────────────────────────────────────────────────────────┤
│  1. 初始化: Rank 0 广播参数到所有 ranks                  │
│  2. Forward: 每个 rank 独立计算 (不同数据)               │
│  3. Backward: 计算本地梯度                               │
│  4. AllReduce: 同步梯度 (平均)                           │
│  5. Update: 每个 rank 独立更新参数 (结果相同)            │
└─────────────────────────────────────────────────────────┘
```

#### 4.2 Individual Parameters DDP

```python
class DDPIndividualParameters(nn.Module):
    """
    最简单的 DDP: 每个参数单独做 AllReduce
    
    缺点: 启动开销大 (每个参数一次 AllReduce)
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        
        # 1. 广播参数从 Rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # 2. 注册 hook: backward 时自动同步梯度
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_allreduce_hook(param))
    
    def _make_allreduce_hook(self, param):
        def hook(grad):
            handle = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
            self._handles.append((param, handle))
        return hook
    
    def finish_gradient_synchronization(self):
        """等待所有 AllReduce 完成"""
        for param, handle in self._handles:
            handle.wait()
        self._handles.clear()
```

#### 4.3 Bucketed DDP (更高效)

```python
class DDPBucketed(nn.Module):
    """
    分桶 DDP: 将参数分组到桶中，桶满了才做 AllReduce
    
    优点: 减少通信启动开销，可以与计算重叠
    """
    def __init__(self, module, bucket_size_mb=25.0):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        
        # 1. 按 backward 顺序构建桶
        self._build_buckets()
        
        # 2. 广播参数
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # 3. 注册 hooks
        self._register_hooks()
```

---

### Phase 5: Sharded Optimizer (ZeRO) (2-3小时)

**概念**: 将优化器状态分片到多个 ranks，减少内存

```
┌────────────────────────────────────────────────────────────┐
│                    ZeRO Stage 1                            │
├────────────────────────────────────────────────────────────┤
│  传统: 每个 rank 存储完整优化器状态 (2x 参数大小 for Adam)  │
│  ZeRO-1: 每个 rank 只存储 1/N 的优化器状态                 │
│         AllGather 在 optimizer.step() 之后收集更新的参数   │
└────────────────────────────────────────────────────────────┘
```

---

## 🔗 与 vLLM 的连接

### FlashAttention → vLLM Attention Backends

```
cs336 实现                          vLLM 实现
─────────────────────────────────────────────────────────
FlashAttentionPyTorch        →     vllm/attention/backends/abstract.py
FlashAttentionTriton         →     vllm/attention/backends/flash_attn.py
                             →     vllm/attention/ops/triton_decode_attention.py
```

**关键代码路径**:
```python
# vLLM 中的 attention 调用链
vllm/model_executor/models/qwen3.py
  └── Qwen3Attention.forward()
        └── self.attn.forward()  # AttentionImpl
              └── FlashAttentionBackend.forward()
                    └── flash_attn_varlen_func()  # FlashAttention 库调用
```

### DDP → vLLM 分布式推理

vLLM 主要使用 **Tensor Parallelism** 而不是 DDP，但理解 DDP 有助于理解分布式通信原语。

```
cs336 DDP 概念                      vLLM 分布式
─────────────────────────────────────────────────────────
AllReduce                    →     vllm/distributed/parallel_state.py
参数广播                      →     tensor_model_parallel_all_reduce()
梯度同步                      →     (推理不需要，但理解有帮助)
```

---

## 📅 学习时间安排

| 阶段 | 预计时间 | 产出 |
|------|---------|------|
| Phase 1: 基础代码阅读 | 1小时 | 理解参考实现 |
| Phase 2: FlashAttention 理论 | 2小时 | 笔记、算法理解 |
| Phase 3: FlashAttention 实现 | 4小时 | 通过 test_attention.py |
| Phase 4: DDP 实现 | 4小时 | 通过 test_ddp.py |
| Phase 5: Sharded Optimizer | 3小时 | 通过 test_sharded_optimizer.py |
| **对照 vLLM 源码** | 2小时 | 深入理解 |

**总计**: ~16小时

---

## 🧪 测试命令

```bash
cd D:\projects\vllm\cs336\assignment2-systems

# 安装依赖
pip install uv
uv sync

# 运行所有测试 (开始时都会失败)
uv run pytest tests/

# 单独测试各模块
uv run pytest tests/test_attention.py -v           # FlashAttention
uv run pytest tests/test_ddp_individual_parameters.py -v  # DDP Individual
uv run pytest tests/test_ddp.py -v                 # DDP Bucketed
uv run pytest tests/test_sharded_optimizer.py -v   # Sharded Optimizer
```

---

## 📚 配套讲座

按顺序学习这些讲座:

1. **Lecture 5 - GPUs** (`nonexecutable/2025 Lecture 5 - GPUs.pdf`)
   - GPU 架构、内存层级、SRAM vs HBM
   
2. **Lecture 6 - Optimization** (`spring2025-lectures/lecture_06.py`)
   - 内存优化、计算优化技术

3. **Lecture 7 - Parallelism** (`nonexecutable/2025 Lecture 7 - Parallelism basics.pdf`)
   - 数据并行、模型并行、流水线并行

4. **Lecture 8 - Distributed Training** (`spring2025-lectures/lecture_08.py`)
   - 分布式训练实战

---

## ✅ 下一步行动

1. [ ] 阅读 `cs336_basics/model.py` 中的 `scaled_dot_product_attention`
2. [ ] 阅读 FlashAttention 论文 (至少 Algorithm 1 & 2)
3. [ ] 创建 `cs336_systems/flash_attention.py` 开始实现
4. [ ] 运行 `uv run pytest tests/test_attention.py -v` 验证实现

祝学习顺利！ 🚀
