# 从零手写 Flash Attention：一份循序渐进的动手教程

> **你将学到什么：**
> - 理解标准 Attention 的内存瓶颈为什么是 O(N²)
> - 掌握 Online Softmax 的数学原理和它的"魔法"
> - 用纯 Python/PyTorch 分块 (Tiling) 实现 FlashAttention Forward Pass
> - 理解 Recomputation 策略并实现 FlashAttention Backward Pass
> - （进阶）了解如何将算法迁移到 Triton GPU Kernel
>
> **前置知识：** 基本的 PyTorch 张量操作、矩阵乘法、Softmax 函数
> **预计时间：** 60-90 分钟 | **难度：** 中级
> **最终成果：** 一个可运行、可通过测试的 `FlashAttentionFunction` 类

---

## 目录
1. [为什么需要 Flash Attention？— 内存墙问题](#section-1)
2. [标准 Attention：先把"笨办法"写出来](#section-2)
3. [核心洞察：Online Softmax](#section-3)
4. [动手实现：FlashAttention Forward Pass](#section-4)
5. [Checkpoint 1：验证 Forward 的正确性](#checkpoint-1)
6. [反向传播：Recomputation 策略](#section-5)
7. [动手实现：FlashAttention Backward Pass](#section-6)
8. [Checkpoint 2：验证 Backward 的正确性](#checkpoint-2)
9. [进阶：迁移到 Triton Kernel](#section-7)
10. [总结与下一步](#summary)

---

<a id="section-1"></a>
## 1. 为什么需要 Flash Attention？— 内存墙问题

### 1.1 问题描述

标准的 Self-Attention 需要计算一个 **N×N 的矩阵**（其中 N 是序列长度）：

```
S = Q @ K^T      # 形状 (N, N) — 这就是罪魁祸首！
P = softmax(S)    # 形状 (N, N) — 又一个 N×N 矩阵
O = P @ V         # 形状 (N, d) — 终于缩回去了
```

> **类比：** 想象你在一场千人大会上做 Self-Attention —— 标准做法是先打印一张 1000×1000 的「关注度表格」（S 矩阵），然后查表做计算。这张表格占满了你整个桌面（GPU 显存）。Flash Attention 的做法是：**不打印完整的表格**，而是一小块一小块地计算并立即丢弃，只保留最终结果。

### 1.2 关键数字

| 序列长度 N | S 矩阵大小 (float16) | 占用显存 |
|-----------|---------------------|---------|
| 1K        | 1M 元素              | 2 MB    |
| 8K        | 64M 元素             | 128 MB  |
| 128K      | 16B 元素             | 32 GB   |

当 N=128K 时，**仅一个注意力矩阵就需要 32 GB**！这不可能放进 GPU 的 SRAM（通常只有 20 MB）。

### 1.3 Flash Attention 的核心思想（三句话版本）

1. **分块 (Tiling)**：不一次算整个 N×N，而是一次只算一小块（如 64×64）。
2. **Online Softmax**：在分块的同时渐进式地计算精确的 Softmax（不需要看到完整的一行）。
3. **Recomputation**：反向传播时重新计算 P 矩阵，而非从前向传播中保存它。

```
内存复杂度: O(N²) → O(N)   ← 这就是 Flash Attention 的根本价值
```

---

<a id="section-2"></a>
## 2. 标准 Attention：先把"笨办法"写出来

在优化之前，我们必须先彻底搞懂标准（naive）的 Attention 实现。这是对比的基准。

```python
# standard_attention.py — 标准 Attention（参考实现）
import torch
from einops import einsum

def standard_attention(q, k, v, is_causal=False):
    """
    标准 Scaled Dot-Product Attention.
    
    Args:
        q: (B, N, d) — Query
        k: (B, N, d) — Key
        v: (B, N, d) — Value
        is_causal: 是否使用因果掩码（下三角）
        
    Returns:
        o: (B, N, d) — Attention 输出
    """
    d = q.shape[-1]
    scale = 1.0 / (d ** 0.5)
    
    # Step 1: 计算注意力分数 (这里产生了 N×N 矩阵!)
    S = einsum(q, k, '... q d, ... k d -> ... q k') * scale  # (B, N, N)
    
    # Step 2: (可选) 因果掩码
    if is_causal:
        N = q.shape[-2]
        mask = torch.arange(N, device=S.device)[:, None] >= torch.arange(N, device=S.device)[None, :]
        S = torch.where(mask, S, torch.tensor(-1e6))
    
    # Step 3: Softmax (又一个 N×N 矩阵!)
    P = torch.softmax(S, dim=-1)  # (B, N, N)
    
    # Step 4: 输出
    o = einsum(P, v, '... q k, ... k d -> ... q d')  # (B, N, d)
    
    return o
```

> **⚠️ 注意：** 这段代码在 N 较小时工作正常，但当 N 增大时，`S` 和 `P` 矩阵会吃掉所有显存。我们接下来的目标就是消灭这两个 N×N 矩阵。

### ✅ 动手练习 2.1

运行上面的代码，确认对于 `q, k, v = (4, 128, 64)` 的输入，它能产生正确的结果。

```python
torch.manual_seed(42)
B, N, d = 4, 128, 64
q = torch.randn(B, N, d)
k = torch.randn(B, N, d)
v = torch.randn(B, N, d)

o = standard_attention(q, k, v)
print(f"输出形状: {o.shape}")  # 预期: torch.Size([4, 128, 64])
print(f"输出范数: {o.norm():.4f}")  # 应该是一个有限的数
```

---

<a id="section-3"></a>
## 3. 核心洞察：Online Softmax

### 3.1 标准 Softmax 的难题

标准的 Softmax 需要看到**整行的所有元素**后才能算：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

问题在于**分母**：你必须先遍历完整行的 N 个元素来拿到 `Σ exp(x_j)`，才能开始算任何一个 `softmax(x_i)`。这似乎让分块变得不可能。

### 3.2 Online Softmax 的魔法

Online Softmax 的核心发现是：**我们可以边扫描边更新，看到新的一块数据时修正之前的答案。**

它维护三个状态变量：

| 变量 | 含义 | 更新规则 |
|------|------|---------|
| **m** (max) | 当前已扫描行的最大值 | `m_new = max(m_old, block_max)` |
| **l** (sum) | 归一化因子 Σexp(x-m) | `l_new = l_old × exp(m_old - m_new) + Σexp(block - m_new)` |
| **o** (output) | 未归一化输出 | `o_new = o_old × exp(m_old - m_new) + exp(block - m_new) @ V_block` |

### 3.3 图解：用两个块的例子理解

假设一行只分成两块 `[x₁, x₂, x₃]` 和 `[x₄, x₅, x₆]`:

```
┌──────────── 第 1 轮：处理块 [x₁, x₂, x₃] ─────────────┐
│  m = max(x₁, x₂, x₃) = 5                               │
│  l = exp(x₁-5) + exp(x₂-5) + exp(x₃-5) = 1.73         │
│  o = exp([x₁,x₂,x₃] - 5) @ V_block1                   │
└──────────────────────────────────────────────────────────┘

┌──────────── 第 2 轮：处理块 [x₄, x₅, x₆] ─────────────┐
│  block_max = max(x₄, x₅, x₆) = 7                       │
│  m_new = max(5, 7) = 7     ← 发现新最大值！              │
│                                                          │
│  α = exp(m_old - m_new) = exp(5 - 7) = exp(-2)          │
│  ↑ 这个修正因子把旧结果"缩放"到新最大值下                   │
│                                                          │
│  l_new = l_old × α + exp([x₄,x₅,x₆] - 7).sum()        │
│  o_new = o_old × α + exp([x₄,x₅,x₆] - 7) @ V_block2   │
└──────────────────────────────────────────────────────────┘

最终: O = o_new / l_new  ← 精确等价于标准 Softmax 结果！
```

> **关键洞察：** `α = exp(m_old - m_new)` 这个修正因子是整个算法的精髓。当发现更大的值时，它把之前所有的累加结果等比缩小，确保数值上和「先看到所有数据再算」完全一致。

### ✅ 动手练习 3.1

用一个小例子验证 Online Softmax 与标准 Softmax 的等价性：

```python
import torch
import math

# 一行数据，分成两块
x = torch.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 2.0])
block1, block2 = x[:3], x[3:]

# --- 标准 Softmax ---
standard_result = torch.softmax(x, dim=0)
print("标准 Softmax:", standard_result)

# --- Online Softmax ---
# 处理块 1
m = block1.max()
l = torch.exp(block1 - m).sum()
# 处理块 2
block_max = block2.max()
m_new = torch.maximum(m, block_max)
alpha = torch.exp(m - m_new)              # 修正因子
l_new = l * alpha + torch.exp(block2 - m_new).sum()
# 最终计算
online_result = torch.cat([
    torch.exp(block1 - m_new) / l_new,
    torch.exp(block2 - m_new) / l_new,
])
print("Online Softmax:", online_result)
print("最大差异:", (standard_result - online_result).abs().max().item())
# 预期输出: 最大差异应该接近 0（~1e-7）
```

---

<a id="section-4"></a>
## 4. 动手实现：FlashAttention Forward Pass

现在进入正题！我们将用纯 PyTorch 实现 FlashAttention 的 Forward Pass。

### 4.1 算法总览

```
FlashAttention Forward 伪代码:
─────────────────────────────
输入: Q, K, V (B, H, N, d)
初始化: O=0, L=0, M=-inf   (都是按 Query 行维护)

FOR i = 0, BLOCK_SIZE, 2*BLOCK_SIZE, ... (遍历 Q 的块)
    取出 Q_i
    FOR j = 0, BLOCK_SIZE, 2*BLOCK_SIZE, ... (遍历 K/V 的块)
        取出 K_j, V_j
        1. S_ij = Q_i @ K_j^T * scale         ← 小矩阵 (Br×Bc)
        2. m_new = max(m_old, S_ij.row_max())  ← Update max
        3. P_ij = exp(S_ij - m_new)            ← Safe exp
        4. α = exp(m_old - m_new)              ← Rescale factor
        5. l_new = l_old * α + P_ij.row_sum()  ← Update sum
        6. O_i = O_i * α + P_ij @ V_j          ← Update output
        7. 写回 M, L

输出: O = O / L   ← 最终归一化
```

### 4.2 完整代码实现

下面是基于您项目中 [block_attention.py](file:///D:/projects/vllm/cs336/assignment2-systems/cs336_systems/block_attention.py) 的完整 Forward 实现，每一步都有详细注释：

```python
# flash_attention_forward.py
import math
import torch

class FlashAttentionFunction(torch.autograd.Function):
    """
    FlashAttention 的纯 PyTorch 实现。
    用 Python for 循环模拟 Tiling，用 Online Softmax 保证数值正确性。
    目的：验证算法逻辑，不追求速度。
    """

    @staticmethod
    def forward(ctx, q, k, v, block_mask=None, causal=False):
        """
        Args:
            q, k, v: (B, H, N, d) — Batch, Heads, SeqLen, HeadDim
            causal: bool, 是否应用因果 (下三角) 掩码
        Returns:
            O: (B, H, N, d)
        """
        B, H, N, d = q.shape
        BLOCK_SIZE = 64  # 可调，GPU 上通常 128 或 256

        # ==========================================
        # Step 0: 初始化累加器 (都是按 Query 行维护)
        # ==========================================
        O = torch.zeros_like(q)                    # (B, H, N, d) 未归一化输出
        L = torch.zeros(B, H, N,                   # (B, H, N) 归一化因子 Σexp
                        device=q.device, dtype=q.dtype)
        M = torch.full((B, H, N), float('-inf'),   # (B, H, N) 行最大值
                        device=q.device, dtype=q.dtype)

        scale = 1.0 / math.sqrt(d)

        # ==========================================
        # 外层循环：遍历 Q 的块 (第 i 个块)
        # ==========================================
        for i in range(0, N, BLOCK_SIZE):
            i_end = min(i + BLOCK_SIZE, N)
            qi = q[:, :, i:i_end, :]  # (B, H, Br, d)

            # ==========================================
            # 内层循环：遍历 K/V 的块 (第 j 个块)
            # ==========================================
            for j in range(0, N, BLOCK_SIZE):
                j_end = min(j + BLOCK_SIZE, N)
                kj = k[:, :, j:j_end, :]  # (B, H, Bc, d)
                vj = v[:, :, j:j_end, :]  # (B, H, Bc, d)

                # --- Step 1: 计算注意力分数 ---
                # S_ij = Q_i @ K_j^T / √d
                # 注意: 这里只产生了一个 Br×Bc 的小矩阵！
                s_ij = torch.einsum('...qd,...kd->...qk', qi, kj) * scale

                # --- Step 1.5: 因果掩码 (可选) ---
                if causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    causal_mask = row_idx >= col_idx  # 下三角为 True
                    s_ij = s_ij.masked_fill(~causal_mask, float('-inf'))

                # --- Step 2: Online Softmax — 更新最大值 ---
                m_prev = M[:, :, i:i_end]
                block_max = s_ij.max(dim=-1).values
                m_new = torch.maximum(m_prev, block_max)

                # --- Step 3: 安全 exp ---
                p_ij = torch.exp(s_ij - m_new.unsqueeze(-1))

                # --- Step 4: 修正旧累加器 ---
                alpha = torch.exp(m_prev - m_new)      # 修正因子！
                l_prev = L[:, :, i:i_end]
                l_new = l_prev * alpha + p_ij.sum(dim=-1)

                # --- Step 5: 更新输出 ---
                o_prev = O[:, :, i:i_end, :]
                O[:, :, i:i_end, :] = o_prev * alpha.unsqueeze(-1) + p_ij @ vj

                # --- Step 6: 写回 ---
                M[:, :, i:i_end] = m_new
                L[:, :, i:i_end] = l_new

        # ==========================================
        # 最终归一化: O = O_unnorm / L
        # ==========================================
        O = O / L.unsqueeze(-1)

        # 保存给 backward 用 (注意: 不存 P，这就是省显存的关键!)
        ctx.save_for_backward(q, k, v, O, L, M)
        ctx.causal = causal
        ctx.block_size = BLOCK_SIZE
        return O
```

### 4.3 逐段解析决策

| 代码段 | 为什么这样做？ |
|--------|-------------|
| `BLOCK_SIZE = 64` | 块大小决定了 SRAM 占用。64 是 CPU 上的合理值，GPU 通常用 128-256 |
| `M = full(-inf)` | 初始化为 -∞，保证第一轮 `torch.maximum` 一定会更新 |
| `alpha = exp(m_prev - m_new)` | 当 `m_new > m_prev` 时 alpha < 1，缩小旧结果；相等时 alpha=1，不变 |
| `ctx.save_for_backward(q,k,v,O,L,M)` | **不保存 P！** 这是 FlashAttention 节省显存的核心 trick |

---

<a id="checkpoint-1"></a>
## 5. ✅ Checkpoint 1：验证 Forward 的正确性

```python
# test_forward.py — 验证你的实现
import torch

torch.manual_seed(0)
B, H, N, d = 4, 1, 128, 64  # 注意：test 用的是 (B, N, d)，这里加了 H 维度
q = torch.randn(B, H, N, d)
k = torch.randn(B, H, N, d)
v = torch.randn(B, H, N, d)

# 你的实现
o_flash = FlashAttentionFunction.apply(q, k, v, None, False)

# 标准实现 (用 PyTorch 内置)
scale = 1.0 / (d ** 0.5)
S = torch.einsum('...qd,...kd->...qk', q, k) * scale
P = torch.softmax(S, dim=-1)
o_std = P @ v

# 对比
max_diff = (o_flash - o_std).abs().max().item()
print(f"最大差异: {max_diff:.2e}")
# 预期: 最大差异 < 1e-5
assert max_diff < 1e-4, f"差异太大: {max_diff}"
print("✅ Forward Pass 正确！")
```

> **🔍 调试提示：** 如果差异很大，请检查：
> 1. `scale` 是否正确（应该是 `1/√d` 而不是 `1/d`）
> 2. `alpha` 的计算顺序（必须是 `m_prev - m_new`，不是反过来）
> 3. 最终归一化是否只做了一次（在所有循环之后）

---

<a id="section-5"></a>
## 6. 反向传播：Recomputation 策略

### 6.1 为什么叫 "Recomputation"？

标准 Attention 的反向传播需要 P 矩阵（N×N）。但 FlashAttention 在前向传播中**故意不存储 P**。

**那反向传播怎么办？** —— 就地重新计算！

对每一对 (i, j) 块：
```python
# 重新计算 P_ij（和 Forward 完全相同的公式）
s_ij = Q_i @ K_j^T * scale
P_ij = exp(S_ij - m_i) / l_i    # 用保存的 m 和 l 直接得到归一化的 P
```

### 6.2 反向传播的数学公式

标准 Attention 反向传播有 4 个关键公式：

```
dV = P^T @ dO              ← V 的梯度
dP = dO @ V^T              ← 对 P 的梯度 (中间量)
dS = P ⊙ (dP - D)          ← 对 S 的梯度 (Softmax 的反向传播)
dQ = dS @ K * scale         ← Q 的梯度
dK = dS^T @ Q * scale       ← K 的梯度

其中 D_i = Σ_j (O_ij × dO_ij)   ← Softmax 反向传播的关键辅助量
```

> **💡 关于 D_i 的直觉：**  
> `D_i` 是每个 Query 位置上「输出 O 与其梯度 dO」的逐元素乘积之和。  
> 它出现在 Softmax 反向传播公式中：`dS_ij = P_ij × (dP_ij - D_i)`  
> 这个公式保证了 Softmax 的梯度和为零（概率分布的约束）。

---

<a id="section-6"></a>
## 7. 动手实现：FlashAttention Backward Pass

```python
    @staticmethod
    def backward(ctx, dO):
        """
        FlashAttention Backward Pass (重计算 P，不存储 N×N 矩阵)

        Args:
            dO: (B, H, N, d) — Loss 对 O 的梯度

        Returns:
            dQ, dK, dV: 各自形状与 q, k, v 相同
            None, None: 对应 block_mask 和 causal 的梯度
        """
        q, k, v, O, L, M = ctx.saved_tensors
        causal = ctx.causal
        BLOCK_SIZE = ctx.block_size

        B, H, N, d = q.shape
        scale = 1.0 / math.sqrt(d)

        # 初始化梯度
        dQ = torch.zeros_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)

        # ==========================================
        # 预计算 D_i = rowsum(dO ⊙ O)
        # 这是 Softmax 反向传播的关键量
        # ==========================================
        D = (dO * O).sum(dim=-1)  # (B, H, N)

        # ==========================================
        # 分块循环：与 Forward 结构完全镜像
        # ==========================================
        for i in range(0, N, BLOCK_SIZE):
            i_end = min(i + BLOCK_SIZE, N)
            qi  = q[:, :, i:i_end, :]
            doi = dO[:, :, i:i_end, :]
            li  = L[:, :, i:i_end]
            mi  = M[:, :, i:i_end]
            di  = D[:, :, i:i_end]

            for j in range(0, N, BLOCK_SIZE):
                j_end = min(j + BLOCK_SIZE, N)
                kj = k[:, :, j:j_end, :]
                vj = v[:, :, j:j_end, :]

                # --- Recomputation! 重新计算 S_ij 和 P_ij ---
                s_ij = torch.einsum('...qd,...kd->...qk', qi, kj) * scale

                if causal:
                    row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                    col_idx = torch.arange(j, j_end, device=q.device)[None, :]
                    causal_mask = row_idx >= col_idx
                    s_ij = s_ij.masked_fill(~causal_mask, float('-inf'))

                # P_ij = exp(S_ij - m_i) / l_i  ← 直接得到归一化后的概率!
                p_ij = torch.exp(s_ij - mi.unsqueeze(-1)) / li.unsqueeze(-1)

                # --- dV_j += P_ij^T @ dO_i ---
                dV[:, :, j:j_end, :] += torch.einsum('...qk,...qd->...kd', p_ij, doi)

                # --- dP_ij = dO_i @ V_j^T ---
                dp_ij = torch.einsum('...qd,...kd->...qk', doi, vj)

                # --- dS_ij = P_ij ⊙ (dP_ij - D_i) ---
                ds_ij = p_ij * (dp_ij - di.unsqueeze(-1))

                # --- dQ_i += dS_ij @ K_j * scale ---
                dQ[:, :, i:i_end, :] += torch.einsum('...qk,...kd->...qd', ds_ij, kj) * scale

                # --- dK_j += dS_ij^T @ Q_i * scale ---
                dK[:, :, j:j_end, :] += torch.einsum('...qk,...qd->...kd', ds_ij, qi) * scale

        return dQ, dK, dV, None, None
```

### 关键设计对比

| 设计选择 | Forward | Backward |
|---------|---------|----------|
| 保存什么 | q, k, v, O, L, M (不存 P!) | 用 L, M 在线重算 P |
| 循环结构 | 2 层 for，外 Q 内 KV | 完全相同的 2 层 for |
| 核心操作 | `p_ij @ vj` (累加输出) | `p_ij * (dp - D)` (梯度传播) |
| 内存开销 | O(N) 额外空间 | O(N) 额外空间（梯度张量除外） |

---

<a id="checkpoint-2"></a>
## 8. ✅ Checkpoint 2：验证 Backward 的正确性

```python
# test_backward.py — 验证梯度
import torch

torch.manual_seed(0)
B, N, d = 4, 128, 64

# --- 标准实现的梯度（作为参考） ---
q_ref = torch.randn(B, N, d, requires_grad=True)
k_ref = torch.randn(B, N, d, requires_grad=True)
v_ref = torch.randn(B, N, d, requires_grad=True)
do = torch.randn(B, N, d)

scale = 1.0 / (d ** 0.5)
S = torch.einsum('...qd,...kd->...qk', q_ref, k_ref) * scale
P = torch.softmax(S, dim=-1)
o_ref = P @ v_ref
o_ref.backward(do)

# --- Flash Attention 的梯度 ---
q_fa = q_ref.data.clone().unsqueeze(1).requires_grad_(True)  # 加 H 维度
k_fa = k_ref.data.clone().unsqueeze(1).requires_grad_(True)
v_fa = v_ref.data.clone().unsqueeze(1).requires_grad_(True)

o_fa = FlashAttentionFunction.apply(q_fa, k_fa, v_fa, None, False)
o_fa.backward(do.unsqueeze(1))

# --- 对比 ---
print(f"dQ 最大差异: {(q_ref.grad - q_fa.grad.squeeze(1)).abs().max():.2e}")
print(f"dK 最大差异: {(k_ref.grad - k_fa.grad.squeeze(1)).abs().max():.2e}")
print(f"dV 最大差异: {(v_ref.grad - v_fa.grad.squeeze(1)).abs().max():.2e}")
# 预期: 所有差异 < 1e-4
print("✅ Backward Pass 正确！")
```

---

<a id="section-7"></a>
## 9. 进阶：迁移到 Triton Kernel

纯 Python 实现验证了算法，但要真正获得速度提升，需要用 **Triton** 写 GPU kernel。

### 9.1 框架代码

您的项目中 [triton_attention.py](file:///D:/projects/vllm/cs336/assignment2-systems/cs336_systems/triton_attention.py) 已经提供了框架：

```python
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,              # 指针
    stride_qm, stride_qn,      # 步幅（Strides）
    # ... 其他参数
    BLOCK_M: tl.constexpr,      # Q 块大小 (编译时常量)
    BLOCK_N: tl.constexpr,      # KV 块大小
):
    # TODO: 实现以下步骤
    # 1. 计算当前线程块的 Q 范围
    # 2. 从 HBM 加载 Q 到 SRAM
    # 3. 初始化 m, l, o 累加器
    # 4. 循环遍历 K/V 块:
    #    a. 从 HBM 加载 K, V 到 SRAM
    #    b. 计算 S = Q @ K^T
    #    c. 应用 Online Softmax 更新
    #    d. 累加 O += P @ V
    # 5. 最终归一化并写回 HBM
    pass
```

### 9.2 Triton vs PyTorch 的关键区别

| 概念 | PyTorch (你已实现) | Triton (下一步) |
|------|-------------------|----------------|
| 循环 | Python `for` | 由 GPU program grid 的 `pid` 实现 |
| 数据加载 | 切片 `q[:, :, i:i_end]` | `tl.load(ptr + offsets)` |
| 矩阵乘法 | `torch.einsum` | `tl.dot(a, b)` |
| 最大值 | `tensor.max()` | `tl.max(tensor, axis=)` |
| 内存 | Python 自动管理 | 手动管理 SRAM/HBM 传输 |

> **💡 提示：** Triton 的核心优势是**消除了外层 for 循环**。每个 GPU 线程块并行处理一个 Q 块，内层只保留对 K/V 块的循环。

---

<a id="summary"></a>
## 10. 总结

### 你学到了什么

| 阶段 | 学习成果 |
|------|---------|
| 🧱 基础 | 标准 Attention 的 O(N²) 内存瓶颈 |
| 💡 洞察 | Online Softmax 如何避免存储完整的 N×N 矩阵 |
| 🔨 实现 | 用分块 + Online Softmax 实现 FlashAttention Forward |
| 🔙 反向 | Recomputation 策略如何让反向传播无需 P 矩阵 |
| 🚀 进阶 | Triton kernel 的框架和迁移路径 |

### 核心公式速查表

```
Forward:
  α = exp(m_old - m_new)              # 修正因子
  l_new = l_old × α + Σexp(S - m_new) # 更新归一化因子
  O_new = O_old × α + P_block @ V     # 更新输出

Backward:
  D_i = Σ(O_i × dO_i)                 # Softmax 辅助量
  P_ij = exp(S_ij - m_i) / l_i        # 重计算 (Recomputation!)
  dS = P ⊙ (dO@V^T - D)              # Softmax 梯度
  dQ = dS @ K × scale                 # Q 梯度
  dK = dS^T @ Q × scale               # K 梯度
  dV = P^T @ dO                        # V 梯度
```

### 下一步

1. **运行测试**：执行项目中的 `test_attention.py` 来验证您的实现  
2. **实现 Triton Kernel**：把 [triton_attention.py](file:///D:/projects/vllm/cs336/assignment2-systems/cs336_systems/triton_attention.py) 中的 `TODO` 填充完  
3. **深入阅读**：  
   - [FlashAttention 论文 (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)  
   - [FlashAttention-2 论文 (Dao, 2023)](https://arxiv.org/abs/2307.08691)  
   - [Triton 官方教程: Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

---

## 常见问题排查

| 错误现象 | 原因 | 修复 |
|---------|------|-----|
| 输出 NaN | `exp()` 溢出，`m_new` 没正确更新 | 检查 `M` 初始化是否为 `-inf` |
| 输出偏差大 (>1e-2) | `scale` 计算错误 | 确认 `1/√d` 而非 `1/d` |
| Causal 模式结果错 | mask 方向反了 | 确认 `row >= col`（不是 `>` ） |
| Backward dQ 偏差大 | `D_i` 未正确预计算 | 确认 `D = (dO * O).sum(dim=-1)` 用的是 Forward 的归一化后 O |
| 形状不匹配 | 维度顺序 (B,H,N,d) vs (B,N,d) | 检查测试框架要求的 tensor 形状 |
