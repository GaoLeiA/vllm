# 第六章：FlashAttention Backward — 重计算策略

> **你将学到什么：**
> - 理解 Recomputation（重计算）为什么能省显存
> - 掌握 FlashAttention 反向传播的四个公式
> - 理解为什么 dK 和 dV 需要累加而 dQ 不需要
> - 用 PyTorch 验证梯度正确性
> - 实现 Triton Backward kernel 的框架
>
> **前置知识：** 第五章（FlashAttention Forward）
> **预计时间：** 60 分钟
> **最终成果：** PyTorch 验证的 Backward + Triton kernel 框架

---

## 1. 为什么需要 "Recomputation"？

### 1.1 标准 Attention 的反向传播

标准 Attention 的反向传播需要 P 矩阵：

```python
# Forward
S = Q @ K^T / √D          # (B, N, N)
P = softmax(S)             # (B, N, N) ← 这个！
O = P @ V                  # (B, N, D)

# Backward
dV = P^T @ dO              # 需要 P
dQ = ... @ K               # 需要 dS = P * (dP - D)
dK = ... @ Q               # 需要 dS
```

**问题：** P 是 N×N 矩阵，和 S 一样大。如果 Forward 不存 P，Backward 就没法做。

### 1.2 FlashAttention 的策略：不存 P，反向时重算

```
Forward:  不保存 P → 省 N² 显存
Backward: 重新计算 P → 不用额外显存

P_ij_recompute = exp(S_ij - m_i) / l_i
```

**关键洞察：** P 不是任意的，它是由 S 通过 softmax 公式**确定性计算**的。
既然 S = Q @ K^T / √D，而 Q, K 都保存了，那 P 可以原地重新计算！

### 1.3 代价：多了一次 S = Q @ K^T 的计算

| 方法 | 显存 | 计算 |
|------|------|------|
| 标准 Attention | O(N²) (存 P) | 1 次 S = Q@K^T |
| FlashAttention | **O(N)** (不存 P) | **2 次 S = Q@K^T** (Forward + Backward) |

**显存换计算：** 这是 FlashAttention 的核心 tradeoff。

---

## 2. 反向传播公式推导

### 2.1 四个核心公式

```
输入: 前向传播的 Q, K, V, O, L, M
梯度输入: dO (Loss 对 O 的梯度)

输出: dQ, dK, dV

公式:
  D_i = Σⱼ (Oᵢⱼ × dOᵢⱼ)              ← (1) 预计算辅助量
  dV = P^T @ dO                         ← (2) V 的梯度
  dP = dO @ V^T                         ← (3) 中间量
  dS = P ⊙ (dP - D)                     ← (4) S 的梯度
  dQ = dS @ K × scale                   ← (5) Q 的梯度
  dK = dS^T @ Q × scale                 ← (6) K 的梯度
```

### 2.2 公式 (1): 为什么需要 D_i？

```
D_i = (dO × O).sum(dim=-1)
```

这是 Softmax 反向传播的**关键辅助量**。

**直觉理解：**
- Softmax 的梯度公式是 `dS = P * (dP - D)`
- 其中 `D_i = Σⱼ Oᵢⱼ × dOᵢⱼ` 是"输出与其梯度的逐元素乘积之和"
- 这个量保证了 softmax 的梯度满足概率分布的约束（梯度之和为零）

### 2.3 公式 (2)-(6): 数据流图

```
dO ─┬─→ P^T @ dO ──────────────→ dV
    │
    └─→ dO @ V^T ─→ dP
                      │
                      ├─→ dP - D ─→ P * (dP - D) ─→ dS
                      │                                         │
                      │                                         ├─→ dS @ K × scale ─→ dQ
                      │                                         └─→ dS^T @ Q × scale ─→ dK
                      └── (已用 dV 的 K)
```

### 2.4 公式 (4) 的 Softmax 梯度

```
dS = P * (dP - D)

这是 softmax 的已知反向传播公式。推导：
  ∂loss/∂Sᵢⱼ = ∂loss/∂Oᵢⱼ × ∂Oᵢⱼ/∂Sᵢⱼ
  Oᵢⱼ = Σₖ Pᵢₖ × Vₖⱼ
  ∂Oᵢⱼ/∂Sᵢⱼ = Pᵢⱼ - Pᵢⱼ × Dᵢ
             = Pᵢⱼ × (1 - Dᵢ)

简化后: dS = P * (dP - D)
```

---

## 3. PyTorch 实现验证

### 3.1 为什么先用 PyTorch 验证？

写 Triton kernel 之前，先用 PyTorch 实现并验证：

```python
def flash_attention_backward_pytorch(q, k, v, o, l, m, do, causal=False, block_size=64):
    B, N, D = q.shape
    scale = 1.0 / math.sqrt(D)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # 预计算 D
    D_i = (do * o).sum(dim=-1)  # (B, N)

    # 与 forward 完全相同的分块循环
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        qi = q[:, i:i_end, :]
        doi = do[:, i:i_end, :]
        li = l[:, i:i_end].unsqueeze(-1)
        mi = m[:, i:i_end].unsqueeze(-1)
        di = D_i[:, i:i_end].unsqueeze(-1)

        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            kj = k[:, j:j_end, :]
            vj = v[:, j:j_end, :]

            # Recomputation
            s_ij = torch.einsum('...qd,...kd->...qk', qi, kj) * scale

            if causal:
                row_idx = torch.arange(i, i_end)[:, None]
                col_idx = torch.arange(j, j_end)[None, :]
                causal_mask = row_idx >= col_idx
                s_ij = s_ij.masked_fill(~causal_mask, float('-inf'))

            # P_ij = exp(S_ij - m_i) / l_i
            p_ij = torch.exp(s_ij - mi) / li

            # dV_j += P^T @ dO
            dv[:, j:j_end, :] += torch.einsum('...qk,...qd->...kd', p_ij, doi)

            # dP = dO @ V^T
            dp_ij = torch.einsum('...qd,...kd->...qk', doi, vj)

            # dS = P * (dP - D)
            ds_ij = p_ij * (dp_ij - di)

            # dQ += dS @ K * scale
            dq[:, i:i_end, :] += torch.einsum('...qk,...kd->...qd', ds_ij, kj) * scale

            # dK += dS^T @ Q * scale
            dk[:, j:j_end, :] += torch.einsum('...qk,...qd->...kd', ds_ij, qi) * scale

    return dq, dk, dv
```

### 3.2 验证方法

```python
# 1. PyTorch 标准 attention 做反向传播
q = torch.randn(B, N, D, requires_grad=True)
k = torch.randn(B, N, D, requires_grad=True)
v = torch.randn(B, N, D, requires_grad=True)
do = torch.randn(B, N, D)

# 标准 forward
s = torch.einsum('...qd,...kd->...qk', q, k) * scale
p = torch.softmax(s, dim=-1)
o = p @ v
o.backward(do)

dq_std, dk_std, dv_std = q.grad, k.grad, v.grad

# 2. FlashAttention 反向传播
dq_fa, dk_fa, dv_fa = flash_attention_backward_pytorch(q, k, v, o, l, m, do)

# 3. 对比
print(f"dQ diff: {(dq_std - dq_fa).abs().max():.2e}")
print(f"dK diff: {(dk_std - dk_fa).abs().max():.2e}")
print(f"dV diff: {(dv_std - dv_fa).abs().max():.2e}")
# 预期: 所有 < 1e-3
```

---

## 4. Triton Backward Kernel 框架

### 4.1 与 Forward 的对比

| | Forward | Backward |
|---|---------|----------|
| **输入** | Q, K, V | Q, K, V, O, L, M, dO |
| **输出** | O, L | dQ, dK, dV |
| **P 来源** | P = exp(S-M)/L (计算中) | P = exp(S-m)/l (重计算) |
| **累加方式** | O += P@V | dK, dV 需要 += (多个 Q block 共享) |

### 4.2 关键区别：为什么 dK 和 dV 要累加？

```
Forward:
  Block(0): 处理 Q[0:64]   → 写 O[0:64]
  Block(1): 处理 Q[64:128] → 写 O[64:128]
  → 每个 O 位置只被一个 block 写

Backward:
  Block(0): 处理 Q[0:64]   → dK += ..., dV += ...
  Block(1): 处理 Q[64:128] → dK += ..., dV += ...
  → 每个 KV 位置被多个 Q block 用到，需要累加!
```

**类比：**
- Forward: 每个工人负责自己的输出区域
- Backward: 每个工人贡献到公共的 KV 区域，需要汇总

### 4.3 完整框架代码

完整代码见 `06_backward.py` 中的 `_flash_attn_bwd_kernel`。

**TODO 实现清单：**

```
1. 在 wrapper 中预计算 D = (dO * O).sum(dim=-1, keepdim=True)
2. 把 D 作为参数传给 kernel
3. 在 kernel 中:
   a. 加载 Q_i, O_i, L_i, M_i, dO_i
   b. for k_block in range(cdiv(Nk, K_BLOCK)):
      - 加载 K_j, V_j
      - S_ij = tl.dot(Q_i, K_j.T) * scale
      - (causal mask)
      - P_ij = tl.exp(S_ij - M_i) / L_i
      - dP_ij = tl.dot(dO_cast, V_j.T)
      - dS_ij = P_ij * (dP_ij - D_i)
      - dQ_acc += tl.dot(dS_ij, K_j) * scale
      - dV += tl.dot(P_ij.T, dO_i)
      - dK += tl.dot(dS_ij.T, Q_i) * scale
      - K_block_ptr.advance()
   c. 写回 dQ (累加后的结果)
   d. 写回 dK, dV (需要 atomicAdd 或用多个 kernel)
```

### 4.4 难点：dK 和 dV 的累加

**问题：** Triton 的 `tl.store` 是原子写还是累加？

**答案：** `tl.store` 是**覆盖写**，不是累加。

**解决方案：** 有两种方法

1. **分两次 kernel：** 第一次清零 dK, dV，第二次累加
2. **用 `tl.atomic_add`：** Triton 原子的累加操作

推荐方法 1，更简单可靠：

```python
# Wrapper 中:
dK = torch.zeros_like(K)
dV = torch.zeros_like(V)

# Kernel 中用 tl.store 写回
# (因为 dK, dV 初始为 0，store 覆盖 = 累加)
```

---

## 5. 调试指南

### 5.1 最常见的错误

| 错误 | 现象 | 修复 |
|------|------|------|
| D_i 用了未归一化的 O | dQ 偏差大 | 用 `O_normalized = O / L` |
| Causal mask 方向反 | causal 模式结果错 | `row >= col` 不是 `row <= col` |
| P_ij 用 M_new 而非 M_i | 梯度偏差 | 重计算用 forward 保存的 M_i |
| scale 是 1/D 而非 1/√D | 所有梯度偏差 | `scale = 1.0 / sqrt(D)` |
| dK/dV 没有初始化 zeros | 结果随机 | `dK = torch.zeros_like(K)` |

### 5.2 调试流程

```
1. 先用小规模数据 (B=1, N=64, D=32)
2. 对比 PyTorch 标准 attention 的梯度
3. 逐层验证:
   - D_i 是否正确?
   - P_ij 重计算是否正确?
   - dV 是否正确?
   - dQ, dK 是否正确?
4. 逐步增大规模
```

---

## 6. 核心公式总结

```
Backward:
  D = (dO × O).sum(dim=-1, keepdim=True)  ← 预计算

  FOR 每个 Q block i, KV block j:
    S_ij = Q_i @ K_j^T × scale
    P_ij = exp(S_ij - m_i) / l_i          ← 重计算!
    dV_j += P_ij^T @ dO_i
    dP = dO_i @ V_j^T
    dS = P ⊙ (dP - D)
    dQ_i += dS @ K_j × scale
    dK_j += dS^T @ Q_i × scale
```

---

## 7. 练习

### 练习 1：补全 Backward Kernel

`06_backward.py` 中的 `_flash_attn_bwd_kernel` 是框架代码，补全 TODO 部分。

### 练习 2：理解 Recomputation 的代价

```python
# 标准 Attention: 1 次 matmul + 存 P (O(N²) 显存)
# FlashAttention: 2 次 matmul + 不存 P (O(N) 显存)

# 问题: 当 N 很大时，2 次 matmul 的额外计算开销能接受吗?
# 答案: 能! 因为省下的显存可以处理更长的序列，
#       而计算时间是线性的，显存是二次的。
```

### 练习 3：思考 Backward 的 Grid 配置

```python
# Forward: grid = (cdiv(Nq, Q_BLOCK), B)
# Backward: grid = ?

# 答案: 同样是 (cdiv(Nq, Q_BLOCK), B)
# 原因: 每个 block 处理相同的 Q block，循环结构相同
```

---

**运行本章代码：** `python 06_backward.py`
**下一章：** 第七章 — 速查表与资源
