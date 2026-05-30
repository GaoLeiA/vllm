"""
Triton 从零开始教程 — 第六章: FlashAttention Backward Pass
==========================================================

本章目标: 理解并实现 FlashAttention 的反向传播。

核心概念: Recomputation (重计算)
  - 前向传播中我们不保存 P 矩阵 (节省显存)
  - 反向传播时我们"原地重新计算" P 矩阵
  - 这样内存复杂度仍然是 O(N) 而不是 O(N²)

反向传播公式 (标准 Attention):
  dV = P^T @ dO                          ← V 的梯度
  dP = dO @ V^T                           ← 中间量
  dS = P * (dP - D)                       ← Softmax 反向
  dQ = dS @ K * scale                     ← Q 的梯度
  dK = dS^T @ Q * scale                   ← K 的梯度
  其中 D_i = rowsum(dO_i * O_i)           ← Softmax 辅助量

注意: FlashAttention 的反向公式与标准 Attention 完全相同!
区别只在: 我们分块计算，不一次性算整个矩阵。
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 第1步: PyTorch 纯实现验证 (推荐先理解这个)
# ============================================================
"""
在写 Triton kernel 之前，先用 PyTorch 实现一遍反向传播，
确保理解公式。这比直接写 Triton 容易调试得多。
"""


def flash_attention_backward_pytorch(q, k, v, o, l, m, do, causal=False, block_size=64):
    """
    PyTorch 实现的 FlashAttention Backward Pass

    Args:
        q, k, v: 前向传播的输入 (B, N, D)
        o: 前向传播的输出 (B, N, D)
        l: 前向传播保存的 L (归一化因子) (B, N)
        m: 前向传播保存的 M (running max) (B, N)
        do: Loss 对 O 的梯度 (B, N, D)
        causal: 是否因果
        block_size: 分块大小

    Returns:
        dq, dk, dv: 各自形状与 q, k, v 相同
    """
    B, N, D = q.shape
    scale = 1.0 / (D ** 0.5)

    print(f"\n[flash_attention_backward_pytorch] 开始 PyTorch 反向传播计算: B={B}, N={N}, D={D}, block_size={block_size}")

    # 初始化梯度
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # 预计算 D_i = rowsum(dO * O)
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

            # Recomputation! 重新计算 S_ij 和 P_ij
            s_ij = torch.einsum('...qd,...kd->...qk', qi, kj) * scale

            if causal:
                row_idx = torch.arange(i, i_end, device=q.device)[:, None]
                col_idx = torch.arange(j, j_end, device=q.device)[None, :]
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


# ============================================================
# 第2步: 用 PyTorch 验证梯度
# ============================================================
def test_backward_pytorch():
    """对比 PyTorch 标准 attention 的梯度和我们的实现"""
    torch.manual_seed(42)
    B, N, D = 4, 128, 64

    print(f"\n[test_backward_pytorch] 准备测试反向传播: B={B}, N={N}, D={D}")

    q = torch.randn(B, N, D, device="cuda", requires_grad=True)
    k = torch.randn(B, N, D, device="cuda", requires_grad=True)
    v = torch.randn(B, N, D, device="cuda", requires_grad=True)
    do = torch.randn(B, N, D, device="cuda")

    # 标准 Attention
    scale = 1.0 / (D ** 0.5)
    s = torch.einsum('...qd,...kd->...qk', q, k) * scale
    p = torch.softmax(s, dim=-1)
    o = p @ v

    # 反向传播
    o.backward(do)
    dq_std, dk_std, dv_std = q.grad.clone(), k.grad.clone(), v.grad.clone()

    # 需要保存前向中间量
    q2 = q.data.clone().detach().requires_grad_(False)
    k2 = k.data.clone().detach().requires_grad_(False)
    v2 = v.data.clone().detach().requires_grad_(False)

    # 前向 (简化版，不用完整 forward)
    o2 = o.detach()
    m2 = s.max(dim=-1).values
    l2 = torch.logsumexp(s, dim=-1)

    dq_fa, dk_fa, dv_fa = flash_attention_backward_pytorch(
        q2, k2, v2, o2, l2, m2, do, causal=False, block_size=64
    )

    print("=== FlashAttention Backward (PyTorch) 验证 ===\n")
    d_q_diff = (dq_std - dq_fa).abs().max().item()
    d_k_diff = (dk_std - dk_fa).abs().max().item()
    d_v_diff = (dv_std - dv_fa).abs().max().item()

    print(f"  dQ max diff: {d_q_diff:.2e}")
    print(f"  dK max diff: {d_k_diff:.2e}")
    print(f"  dV max diff: {d_v_diff:.2e}")

    if d_q_diff < 1e-3 and d_k_diff < 1e-3 and d_v_diff < 1e-3:
        print("\n  ✅ 所有梯度正确!")
    else:
        print("\n  ⚠️ 梯度有较大差异，请检查公式")


# ============================================================
# 第3步: Triton Backward Kernel (框架)
# ============================================================
"""
Triton Backward Kernel 的设计与 Forward 非常对称:

Forward:
  grid = (num_q_blocks, num_batches)
  每个 block 处理 Q 的一行块

Backward:
  grid = (num_q_blocks, num_batches)
  每个 block 处理 Q 的一行块
  需要计算 dq, dk, dv

关键区别:
  1. 输入多了 dO (梯度)
  2. 需要预计算 D_i = (dO * O).sum(dim=-1)
  3. 三个输出 dQ, dK, dV 需要累加 (+=)
"""


@triton.jit
def _flash_attn_bwd_kernel(
    Q, K, V, O, L, M,       # 前向传播的中间结果
    dO,                       # 对输出的梯度
    dQ, dK, dV,               # 输出: 对输入的梯度
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_lb, stride_ln,
    stride_mb, stride_mn,
    stride_dob, stride_don, stride_dod,
    stride_dqb, stride_dqn, stride_dqd,
    stride_dkb, stride_dkn, stride_dkd,
    stride_dvb, stride_dvn, stride_dvd,
    Nq, Nk, D,
    scale,
    is_causal: tl.constexpr,
    Q_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    """
    FlashAttention Backward Kernel (框架代码)

    TODO 实现步骤:
    1. 确定当前 Q block
    2. 加载 Q block 和 dO block
    3. 预计算 D_i (在 kernel 外部预计算更好)
    4. 循环遍历 KV blocks:
       a. 加载 K_j, V_j
       b. 重新计算 S_ij = Q_i @ K_j^T * scale
       c. 应用 causal mask
       d. 重计算 P_ij = exp(S_ij - m_i) / l_i
       e. 计算 dP_ij = dO_i @ V_j^T
       f. 计算 dS_ij = P_ij * (dP_ij - D_i)
       g. 累加到 dQ, dK, dV
    """

    q_block_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    # ---- 构造 block pointers ----
    Q_block_ptr = tl.make_block_ptr(
        base=Q + batch_id * stride_qb,
        shape=(Nq, D),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + batch_id * stride_kb,
        shape=(Nk, D),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + batch_id * stride_vb,
        shape=(Nk, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(K_BLOCK, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + batch_id * stride_ob,
        shape=(Nq, D),
        strides=(stride_on, stride_od),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        base=L + batch_id * stride_lb,
        shape=(Nq, 1),
        strides=(stride_ln, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1),
        order=(1, 0),
    )

    M_block_ptr = tl.make_block_ptr(
        base=M + batch_id * stride_mb,
        shape=(Nq, 1),
        strides=(stride_mn, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        base=dO + batch_id * stride_dob,
        shape=(Nq, D),
        strides=(stride_don, stride_dod),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    # ---- 加载数据 ----
    Q_i = tl.load(Q_block_ptr)
    O_i = tl.load(O_block_ptr)
    L_i = tl.load(L_block_ptr)
    M_i = tl.load(M_block_ptr)
    dO_i = tl.load(dO_block_ptr)

    # ---- 初始化梯度累加器 ----
    dQ_acc = tl.zeros((Q_BLOCK, D), dtype=tl.float32)
    # dK 和 dV 需要在循环中累加

    # ---- 循环遍历 KV blocks ----
    for k_block_id in range(tl.cdiv(Nk, K_BLOCK)):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        # Recomputation
        S_ij = tl.dot(Q_i, K_j.T) * scale

        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        # 重计算 P_ij
        # P_ij = exp(S_ij - m_i) / l_i
        # 注意: 这里用 m_i (forward 时的 max) 而不是 online softmax 的 M_acc
        p_ij = tl.exp(S_ij - M_i) / L_i

        # dP_ij = dO_i @ V_j^T
        dO_cast = dO_i.to(K_block_ptr.type.element_ty)
        dp_ij = tl.dot(dO_cast, V_j.T)

        # D_i = rowsum(dO * O) — 需要从外部传入或预计算
        # 这里简化: 假设 D_i 已经通过额外参数传入
        # 实际实现中，最好在外面预计算 D_i 然后作为参数传入

        # dS_ij = P_ij * (dP_ij - D_i)
        # dS = p_ij * (dp_ij - di)

        # dQ += dS @ K * scale
        # dK += dS^T @ Q * scale
        # dV += P^T @ dO

        # 移动指针
        K_block_ptr = K_block_ptr.advance((K_BLOCK, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK, 0))

    # ---- 写回 dQ ----
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ + batch_id * stride_dqb,
        shape=(Nq, D),
        strides=(stride_dqn, stride_dqd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )
    # tl.store(dQ_block_ptr, dQ_acc)


# ============================================================
# 第4步: 完整练习 —— 实现 Backward Wrapper
# ============================================================
def flash_attention_backward_full(Q, K, V, O, L, M, dO, is_causal=False):
    """
    完整的反向传播包装函数

    这是你完成的第一个"完整" FlashAttention backward!
    对比上面的框架代码，补全 TODO 部分。

    提示:
    1. D_i 可以在外面预计算: D = (dO * O).sum(dim=-1, keepdim=True)
    2. dK 和 dV 也需要用 block pointers 写回
    3. grid 配置和 forward 一样: (cdiv(Nq, Q_BLOCK), B)
    """
    # TODO: 补全实现
    raise NotImplementedError("练习: 补全反向传播实现")


# ============================================================
# 第5步: 调试指南
# ============================================================
"""
调试 FlashAttention Backward 的常见陷阱:

1. D_i 计算错误
   - 错误: 用了未归一化的 O
   - 正确: 用 forward 输出中归一化后的 O (即 O / L)
   - 公式: D_i = (dO * O_normalized).sum(dim=-1)

2. Causal mask 方向反了
   - 错误: q_idx <= k_idx
   - 正确: q_idx >= k_idx (当前位置只能看到之前的位置)

3. scale 错误
   - 错误: 1/D 而不是 1/sqrt(D)
   - 正确: scale = 1.0 / sqrt(D)

4. P_ij 重计算错误
   - 错误: 用 M_new (online softmax 的合并 max)
   - 正确: 用 M_i (forward 时每行的 max)
   - 公式: P_ij = exp(S_ij - m_i) / l_i

5. 梯度累加 vs 赋值
   - dQ 和 dO 每个 Q block 只处理一次 → 可以直接赋值
   - dK 和 dV 每个 KV block 被多个 Q block 使用 → 必须累加 (+=)
   - 所以 dK 和 dV 需要初始化为 zeros

6. 数据类型
   - 中间计算用 float32 避免溢出
   - 最终写回时转换回输入数据类型
"""


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        test_backward_pytorch()
        print("\n🎉 第六章框架完成!")
        print("\n练习: 补全 _flash_attn_bwd_kernel 的实现")
        print("提示: 参考上面的 PyTorch 实现，逐行翻译成 Triton")
    else:
        print("需要 GPU 才能运行这些测试")
