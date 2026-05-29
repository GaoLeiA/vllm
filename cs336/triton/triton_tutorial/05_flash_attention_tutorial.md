# 第五章：FlashAttention Forward — 核心章节

> **你将学到什么：**
> - 理解标准 Attention 的 O(N²) 内存瓶颈
> - 掌握 Online Softmax 的数学原理
> - 实现完整的 FlashAttention Triton Kernel
> - 理解 Block Pointer 和 `tl.make_block_ptr`
> - 支持 Causal Masking
>
> **前置知识：** 第二章（元素级）+ 第三章（行级聚合）+ 第四章（Tiling）
> **预计时间：** 90 分钟
> **最终成果：** 一个通过 PyTorch 验证的 FlashAttention Triton Kernel

---

## 1. 标准 Attention 的问题

### 1.1 标准 Attention 的计算

```python
# Scaled Dot-Product Attention
S = Q @ K^T / sqrt(D)    # (B, H, N, N)  ← 这里!
P = softmax(S, dim=-1)    # (B, H, N, N)  ← 这里!
O = P @ V                 # (B, H, N, D)
```

### 1.2 内存数字

| 序列长度 N | S 矩阵大小 (fp16) | P 矩阵大小 (fp16) | 总计 |
|-----------|-------------------|-------------------|------|
| 1K | 2 MB | 2 MB | 4 MB |
| 8K | 128 MB | 128 MB | 256 MB |
| 32K | 2 GB | 2 GB | 4 GB |
| 128K | 32 GB | 32 GB | **64 GB** |

**GPU 显存通常只有 24-80 GB。** 当 N=128K 时，仅 Attention 矩阵就需要 64 GB！

### 1.3 FlashAttention 的核心洞察

> **不存储完整的 N×N 矩阵，而是分块计算、即时丢弃。**

```
标准 Attention:    [全部 N×N 矩阵存在显存中]
FlashAttention:    [每次只处理 64×64 小块，用完即丢]
```

内存复杂度从 **O(N²)** 降到 **O(N)**。

---

## 2. Online Softmax — FlashAttention 的数学核心

### 2.1 问题：为什么不能分块算 Softmax？

标准 Softmax:
```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

分母 `Σⱼ exp(xⱼ)` 需要看到**所有元素**才能算。如果分块：
- 第 1 块算到 `l₁ = Σᵢ₁ exp(xᵢ₁)`
- 第 2 块算到 `l₂ = Σᵢ₂ exp(xᵢ₂)`
- 但 `l = l₁ + l₂` 不对！因为**每块的 exp 偏移不同**

### 2.2 Online Softmax 的解决方案

维护三个**状态变量**，每处理一个新块时更新：

| 变量 | 含义 | 初始值 |
|------|------|--------|
| **M** (max) | 已处理部分的最大值 | `-inf` |
| **L** (sum) | 归一化因子 `Σexp(x-m)` | `0` |
| **O** (output) | 未归一化的输出 `ΣP·V` | `0` |

### 2.3 更新公式（最重要的一页！）

处理新块 `(S_block, V_block)` 时：

```
M_block = max(S_block, axis=1)             ← 当前块的 max
M_new   = max(M_old, M_block)              ← 合并后的 max

P_block = exp(S_block - M_block)           ← 安全 exp

L_new   = exp(M_old - M_new) × L_old
          + ΣP_block × exp(M_block - M_new)

O_new   = exp(M_old - M_new) × O_old
          + P_block @ V_block

M = M_new, L = L_new, O = O_new
```

最终结果: `Output = O / L`

### 2.4 图解：一个直观的例子

```
一行数据: [1, 3, 2, 5, 4, 2], 分成两块 [1,3,2] 和 [5,4,2]

第 1 轮: 处理 [1, 3, 2]
  M = max(1,3,2) = 3
  L = exp(1-3) + exp(3-3) + exp(2-3) = 0.135 + 1.0 + 0.368 = 1.503
  O = P₁ @ V₁

第 2 轮: 处理 [5, 4, 2]
  M_block = max(5,4,2) = 5
  M_new = max(3, 5) = 5              ← 发现更大值！

  α₁ = exp(3 - 5) = 0.135            ← 旧结果缩小
  α₂ = exp(3 - 5) = 0.135            ← 旧结果再缩小

  L_new = 1.503 × 0.135 + (exp(0)+exp(-1)+exp(-3))
        = 0.203 + (1.0 + 0.368 + 0.050) = 1.621

  O_new = O × 0.135 + P₂ @ V₂

最终: Output = O_new / L_new
```

**核心洞察：** `exp(M_old - M_new)` 是"修正因子"。当新块有更大值时，之前的所有累加结果等比缩小，确保数值上和"一次性算完"完全一致。

---

## 3. Block Pointer — Triton 的内存访问利器

### 3.1 什么是 Block Pointer？

```python
Q_block_ptr = tl.make_block_ptr(
    base=Q_ptr + batch_id * stride_qb,     # 起始地址
    shape=(Nq, D),                          # 完整形状
    strides=(stride_qn, stride_qd),         # 步幅
    offsets=(q_block_id * Q_BLOCK, 0),      # 当前偏移
    block_shape=(Q_BLOCK, D),               # 加载的形状
    order=(1, 0),                           # 内存排序
)
```

**类比：** 像一个"可移动的取景框" — 你告诉 Triton 你要看哪个位置、多大的一块数据，Triton 负责高效地加载。

### 3.2 为什么用 Block Pointer？

| 方式 | 优点 | 缺点 |
|------|------|------|
| 手动 offset | 灵活 | 容易出错，无法优化 |
| **Block Pointer** | 自动 memory coalescing | 需要理解参数 |

**Memory Coalescing:** 相邻线程访问连续内存地址时，GPU 可以一次加载多个元素。Block Pointer 让 Triton 自动做这个优化。

### 3.3 `advance()` — 移动指针

```python
K_block_ptr = K_block_ptr.advance((K_BLOCK, 0))
```

`advance` 沿着指定方向移动指针。这里 `(K_BLOCK, 0)` 表示沿第 0 维（行）移动 K_BLOCK 个元素。

**在循环中：** 每次迭代 `advance` 一次，正好移动到下一个 KV block。

---

## 4. 完整 Kernel 实现

### 4.1 Kernel 签名

```python
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O, L,           # 指针
    # strides
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,
    # 问题参数
    Nq, Nk, D,
    scale,
    # 编译期常量
    is_causal: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
```

**参数分类：**
- **运行时参数：** 每个 kernel launch 都可以不同（Q, K, V 等）
- **constexpr 参数：** 编译期确定，用于代码生成（Q_BLOCK, K_BLOCK, is_causal）

### 4.2 完整代码

完整代码见 `05_flash_attention.py` 中的 `_flash_attn_fwd_kernel` 函数。

以下是**关键步骤的对照表**：

| 步骤 | 代码片段 | 说明 |
|------|---------|------|
| 1. 确定线程块 | `pid = program_id(0), batch_id = program_id(1)` | 每线程块处理一个 Q block |
| 2. 构造 block ptrs | `tl.make_block_ptr(...)` | Q, K, V, O, L 各一个 |
| 3. 加载 Q block | `Q_i = tl.load(Q_block_ptr)` | 一次性加载 (Q_BLOCK, D) |
| 4. 初始化状态 | `M=-inf, L=0, O=0` | Online Softmax 初始状态 |
| 5. 遍历 KV blocks | `for k_block_id in range(cdiv(Nk, K_BLOCK))` | 内层循环 |
| 5a. 加载 K,V | `K_j = tl.load(K_block_ptr)` | 每次加载 (K_BLOCK, D) |
| 5b. 算 S | `S = tl.dot(Q_i, K_j.T) * scale` | Q_BLOCK × K_BLOCK 小矩阵 |
| 5c. Causal mask | `if is_causal: S = where(mask, S, -1e6)` | 编译期 if |
| 5d. Online Softmax | `M_new = max(M, M_block)` 等 | 更新三个状态 |
| 5e. 更新 O | `O_new = α × O_old + P @ V` | 累加加权输出 |
| 5f. 移动指针 | `K_block_ptr.advance((K_BLOCK, 0))` | 准备下一轮 |
| 6. 归一化 | `O = O / L` | 最终输出 |
| 7. 写回 | `tl.store(O_block_ptr, O)` | 写回 HBM |

### 4.3 Causal Mask 的实现

```python
if is_causal:
    q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
    k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
    causal_mask = q_idx >= k_idx
    S_ij = tl.where(causal_mask, S_ij, -1e6)
```

**原理：**
- `q_idx` 形状 `(Q_BLOCK, 1)` — 每行一个 Query 位置
- `k_idx` 形状 `(1, K_BLOCK)` — 每列一个 Key 位置
- `q_idx >= k_idx` 形状 `(Q_BLOCK, K_BLOCK)` — 下三角为 True
- `tl.where(mask, S, -1e6)` — 非下三角位置设为 -inf

**为什么是 `-1e6` 而不是 `-inf`？**
- `-inf` 在 fp16 中可能导致 NaN
- `-1e6` 的 `exp(-1e6) ≈ 0`，效果等同于 -inf
- 更安全

---

## 5. PyTorch Wrapper

### 5.1 使用 `torch.autograd.Function`

```python
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Q_BLOCK = 64
        K_BLOCK = 64
        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device)

        grid = (triton.cdiv(Nq, Q_BLOCK), B)

        _flash_attn_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            ...
            D=D, Q_BLOCK=Q_BLOCK, K_BLOCK=K_BLOCK, is_causal=is_causal,
        )

        return O
```

### 5.2 Grid 配置

```python
grid = (triton.cdiv(Nq, Q_BLOCK), B)
```

- 第 0 维：`Nq / Q_BLOCK` 个 Q block
- 第 1 维：`B` 个 batch

**总共启动的 kernel 数：** `ceil(Nq / Q_BLOCK) × B`

**经验法则：** 启动的 kernel 数 ≥ 4 × SM 数

```python
# H100: 132 SM → 至少 528 个 kernel
# 如果 Nq=128, Q_BLOCK=64, B=4 → 2×4 = 8 个 kernel ❌
# 如果 Nq=2048, Q_BLOCK=64, B=4 → 32×4 = 128 个 kernel ⚠️
# 如果 Nq=2048, Q_BLOCK=64, B=16 → 32×16 = 512 个 kernel ✅
```

---

## 6. 验证正确性

### 6.1 对比 PyTorch

```python
Q = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
K = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
V = torch.randn(B, N, D, device="cuda", dtype=torch.float16)

# PyTorch 参考
o_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

# Triton 实现
o_triton = FlashAttentionTriton.apply(Q, K, V, is_causal=False)

# 对比
diff = (o_torch.float() - o_triton.float()).abs().max().item()
print(f"max diff: {diff:.4e}")
```

### 6.2 预期误差

```
N=128, D=64:  max_diff ~ 0.1 ~ 1.0  (fp16 精度限制)
N=256, D=128: max_diff ~ 0.5 ~ 2.0
N=512, D=64:  max_diff ~ 1.0 ~ 5.0  (误差随 N 增大)
```

**误差来源：**
1. FP16 精度（只有 ~3 位有效数字）
2. Online Softmax 的累积误差（多轮 `exp(M_old - M_new)` 乘法）
3. Triton 的 `tl.dot` 实现精度

**`max_diff < 10` 通常可以接受。**

---

## 7. 思考题

### Q1：为什么 Q 在循环外加载一次，而 KV 在循环内加载？

A：Q 的大小是 `Q_BLOCK × D`，K/V 每次加载的大小是 `K_BLOCK × D`。
Q 只需要一次（因为每个 KV block 都要和同一个 Q 做 dot），而 KV 需要遍历所有块。
这减少了 Q 的显存读取次数。

### Q2：为什么 `L_block_ptr` 的形状是 `(Nq, 1)` 而不是 `(Nq, D)`？

A：L 是每行的归一化因子，每行只有**一个值**（所有 D 维共享同一个 L）。
所以形状是 `(Nq, 1)`，不需要 D 维。

### Q3：`M_block` 为什么用 `M_block` 而不是 `M_new` 来计算 `P_ij`？

A：这是最容易搞错的地方！
- `P_ij = exp(S_ij - M_block)` — 当前块的 softmax，用当前块 max 偏移
- 后续更新 `L_new` 时，用 `M_new` 做全局修正

如果直接用 `M_new`：
```
P_ij_wrong = exp(S_ij - M_new)
```
当 `M_new > M_block` 时，`exp(S_ij - M_new)` 会更小，导致 P 不正确。

正确的做法是：先用 `M_block` 偏移得到正确的 `P_ij`，再用 `M_new` 修正 `L_acc` 和 `O_acc`。

### Q4：tl.make_block_ptr 的 order=(1, 0) 是什么意思？

A：这告诉 Triton 内存布局的排序。
- `(1, 0)` 表示第 0 维（行）stride 对应顺序 1，第 1 维（列）stride 对应顺序 0
- 这让 Triton 能做更高效的 coalesced access
- 对于行主序的 PyTorch tensor，`order=(1, 0)` 是正确的

---

## 8. 常见错误排查

| 错误 | 原因 | 修复 |
|------|------|------|
| 输出全是 NaN | `M` 初始值不是 `-inf` | 改为 `tl.full((Q_BLOCK, 1), float("-inf"), ...)` |
| 误差很大 (>10) | `alpha` 计算顺序反了 | 确认 `M_acc - M_new` 不是 `M_new - M_acc` |
| causal 结果不对 | mask 方向反了 | `q_idx >= k_idx` 不是 `q_idx <= k_idx` |
| shape mismatch | strides 传错了 | 用 `tensor.stride(0), tensor.stride(1), tensor.stride(2)` |
| kernel 不启动 | grid 配置错误 | 确认 `cdiv(Nq, Q_BLOCK)` 不是 `Nq // Q_BLOCK` |

---

## 9. 核心公式速查

```
FlashAttention Forward (每轮 KV block):

  S = Q @ K^T / √D                    # 小矩阵
  M_block = max(S, axis=1)              # 当前块 max
  M_new = max(M_acc, M_block)           # 合并 max

  P = exp(S - M_block)                  # 安全 exp
  L_new = exp(M_acc-M_new)·L + sum(P)·exp(M_block-M_new)
  O_new = exp(M_acc-M_new)·O + P @ V

  M_acc = M_new, L_acc = L_new, O_acc = O_new

最终: Output = O_acc / L_acc
```

---

## 10. 下一步

本章是整个教程的核心。恭喜你完成了 FlashAttention 的 Triton 实现！

下一章（第六章）我们将实现 **Backward Pass**，包括：
- Recomputation 策略（不保存 P，反向时重新计算）
- 梯度公式的 Triton 实现

**运行本章代码：** `python 05_flash_attention.py`
**下一章：** 第六章 — FlashAttention Backward
