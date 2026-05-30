"""
Triton 从零开始教程 — 第五章: FlashAttention
==========================================

本章目标: 用 Triton 实现完整的 FlashAttention Forward Kernel。

这是本章也是整个教程的核心。我们将:
  1. 回顾标准 Attention 的内存瓶颈
  2. 理解 Online Softmax 的原理
  3. 实现 Tiled FlashAttention
  4. 支持 Causal Masking

前置知识:
  - 第二章 (元素级操作)
  - 第三章 (行级聚合)
  - 第四章 (矩阵乘法 Tiling)
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 第1步: 回顾 —— 标准 Attention 的问题
# ============================================================
"""
标准 Attention:
  S = Q @ K^T / sqrt(d)    # (B, H, Nq, Nk)  ← N² 矩阵!
  P = softmax(S, dim=-1)    # (B, H, Nq, Nk)  ← 又一个 N² 矩阵!
  O = P @ V                 # (B, H, Nq, d)

当序列长度 N = 32768, d = 128:
  S 矩阵大小: 32768 × 32768 × 2 bytes (fp16) = 2 GB
  P 矩阵大小: 同样是 2 GB
  总共: 4 GB 的中间结果!

FlashAttention 的核心思想:
  1. 分块 (Tiling): 每次只处理 Q 的一个小块和 KV 的一个小块
  2. Online Softmax: 边处理边更新，不需要完整 S 矩阵
  3. 内存复杂度: O(N²) → O(N)
"""


# ============================================================
# 第2步: Online Softmax 的数学 (复习)
# ============================================================
"""
传统 Softmax: 需要看到整行所有元素才能计算

Online Softmax: 可以分块增量计算

维护三个状态:
  M: 当前已处理部分的最大值
  L: 当前已处理部分的归一化因子 (sum of exp)
  O: 当前已处理部分的加权和 (未归一化输出)

每处理一个新块 (S_new, V_new):
  M_new = max(M_old, S_new.row_max())
  alpha = exp(M_old - M_new)  ← 修正因子!
  P_new = exp(S_new - M_new)
  L_new = alpha * L_old + P_new.row_sum()
  O_new = alpha * O_old + P_new @ V_new

最终结果: Output = O_new / L_new
"""


# ============================================================
# 第3步: 实现 FlashAttention Triton Kernel
# ============================================================
"""
内核设计:

Grid: (num_q_blocks, num_batches)
  - 每个 block 处理 Q 的一个 Q_BLOCK 行
  - 每个 block 处理一个 batch

Loop: 遍历 KV 的 K_BLOCK 块

每个 block 内部:
  1. 加载 Q block (一次性加载到 SRAM)
  2. 初始化 M=−inf, L=0, O=0
  3. 循环: 对每个 KV block → 做 Online Softmax 更新
  4. 写回最终输出
"""


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O, L,            # 指针
    stride_qb, stride_qn, stride_qd,   # Q 的 strides
    stride_kb, stride_kn, stride_kd,   # K 的 strides
    stride_vb, stride_vn, stride_vd,   # V 的 strides
    stride_ob, stride_on, stride_od,   # O 的 strides
    stride_lb, stride_ln,                    # L 的 strides
    Nq, Nk, D,                       # 序列长度和维度
    scale,                           # 1/sqrt(D)
    is_causal: tl.constexpr,          # 是否因果掩码
    Q_BLOCK: tl.constexpr,            # Q 的块大小 (编译期常量)
    K_BLOCK: tl.constexpr,            # K/V 的块大小
):
    """
    FlashAttention Forward Kernel (Triton 实现)

    每个 kernel 负责:
      - 一个 batch (batch_id)
      - 一个 Q 的块 (q_block_id)

    参数详解:
      stride_qb: Q 的 batch stride (Q[b,i,j] → Q[b*stride_qb + i*stride_qn + j*stride_qd])
      Nq: Query 序列长度
      Nk: Key 序列长度
      D: Head dimension
      scale: 1/sqrt(D)，用于缩放注意力分数
      is_causal: 是否使用 causal masking (解码器场景)
    """

    # ---- Step 1: 确定当前线程块处理哪个 Q block 和哪个 batch ----
    q_block_id = tl.program_id(0)   # 第几个 Q block
    batch_id = tl.program_id(1)     # 第几个 batch

    # ---- Step 2: 构造 Block Pointers ----
    # Triton 的 block pointer 是一个高效的数据加载工具
    # 它告诉 Triton: "从这个位置开始，按这个步幅，加载这个形状的块"

    # Q: (Nq, D) 形状，从 (q_block_id*Q_BLOCK, 0) 开始加载
    Q_block_ptr = tl.make_block_ptr(
        base=Q + batch_id * stride_qb,
        shape=(Nq, D),
        strides=(stride_qn, stride_qd),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),  # 列主序 → 行主序 (Triton 内部优化内存合并访问)
    )

    # K, V: 从第 0 行开始，在循环中用 advance() 移动
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

    # O: 写入结果
    O_block_ptr = tl.make_block_ptr(
        base=O + batch_id * stride_ob,
        shape=(Nq, D),
        strides=(stride_on, stride_od),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, D),
        order=(1, 0),
    )

    # L: 记录 logsumexp (用于 backward, 这里我们只实现 forward)
    L_block_ptr = tl.make_block_ptr(
        base=L + batch_id * stride_lb,
        shape=(Nq, 1),
        strides=(stride_ln, 1),
        offsets=(q_block_id * Q_BLOCK, 0),
        block_shape=(Q_BLOCK, 1),
        order=(1, 0),
    )

    # ---- Step 3: 加载 Q block 到 SRAM ----
    Q_i = tl.load(Q_block_ptr)  # shape: (Q_BLOCK, D)

    # ---- Step 4: 初始化 Online Softmax 状态 ----
    # M: running max of attention scores (每行一个值)
    # L: running sum of exp scores (每行一个值)
    # O: running weighted sum of V (每行 D 个值)
    M_acc = tl.full((Q_BLOCK, 1), float("-inf"), dtype=tl.float32)
    L_acc = tl.zeros((Q_BLOCK, 1), dtype=tl.float32)
    O_acc = tl.zeros((Q_BLOCK, D), dtype=tl.float32)

    # ---- Step 5: 遍历 KV blocks ----
    for k_block_id in range(tl.cdiv(Nk, K_BLOCK)):
        # 加载 K 和 V blocks
        K_j = tl.load(K_block_ptr)  # (K_BLOCK, D)
        V_j = tl.load(V_block_ptr)  # (K_BLOCK, D)

        # ---- Step 5.1: S_ij = Q_i @ K_j^T / sqrt(D) ----
        # 这是一个 Q_BLOCK x K_BLOCK 的小矩阵
        S_ij = tl.dot(Q_i, K_j.T) * scale  # (Q_BLOCK, K_BLOCK)

        # ---- Step 5.2: Causal Mask ----
        # 解码器场景: Q 的第 i 个位置只能看到 K 的第 j<=i 个位置
        if is_causal:
            q_idx = q_block_id * Q_BLOCK + tl.arange(0, Q_BLOCK)[:, None]
            k_idx = k_block_id * K_BLOCK + tl.arange(0, K_BLOCK)[None, :]
            causal_mask = q_idx >= k_idx
            S_ij = tl.where(causal_mask, S_ij, -1e6)

        # ---- Step 5.3: Online Softmax ----
        # 找当前块的 max (用于数值稳定)
        M_block = tl.max(S_ij, axis=1, keep_dims=True)  # (Q_BLOCK, 1)

        # 更新 running max
        M_new = tl.maximum(M_acc, M_block)  # (Q_BLOCK, 1)

        # 计算 P_ij = exp(S_ij - M_block) (注意: 用 M_block 不是 M_new!)
        P_ij = tl.exp(S_ij - M_block)  # (Q_BLOCK, K_BLOCK)

        # 更新 running sum (归一化因子)
        # 关键公式: L_new = exp(M_old - M_new) * L_old + sum(exp(S_block - M_new))
        L_new = (
            tl.exp(M_acc - M_new) * L_acc +
            tl.sum(P_ij, axis=1, keep_dims=True) * tl.exp(M_block - M_new)
        )

        # 更新输出 (加权累加)
        # O_new = exp(M_old - M_new) * O_old + P_block @ V_block
        P_cast = P_ij.to(V_block_ptr.type.element_ty)  # 类型转换
        O_new = (
            tl.exp(M_acc - M_new) * O_acc +
            tl.dot(P_cast, V_j)
        )

        # ---- Step 5.4: 更新 running 状态 ----
        M_acc = M_new
        L_acc = L_new
        O_acc = O_new

        # ---- Step 5.5: 移动 K/V 指针到下一个 block ----
        K_block_ptr = K_block_ptr.advance((K_BLOCK, 0))
        V_block_ptr = V_block_ptr.advance((K_BLOCK, 0))

    # ---- Step 6: 最终归一化 ----
    # Output = O / L
    # LogSumExp = M + log(L)
    O_i = O_acc / L_acc
    L_i = M_acc + tl.log(L_acc)

    # ---- Step 7: 写回 ----
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)


# ============================================================
# 第4步: 包装函数 —— 连接 Python 和 Triton
# ============================================================
class FlashAttentionTriton(torch.autograd.Function):
    """
    PyTorch autograd wrapper for FlashAttention Triton kernel.

    使用方式:
      output = FlashAttentionTriton.apply(Q, K, V, causal)

    输入形状: (B, N, D) — batch, sequence length, head dimension
    输出形状: (B, N, D)
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """前向传播"""
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        B, Nq, D = Q.shape
        Nk = K.shape[1]

        Q_BLOCK = 64
        K_BLOCK = 64
        scale = D ** -0.5

        O = torch.empty_like(Q)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)

        grid = (triton.cdiv(Nq, Q_BLOCK), B)

        print(f"\n[FlashAttentionTriton.forward] 准备启动 Kernel:")
        print(f"  -> B={B}, Nq={Nq}, Nk={Nk}, D={D}, is_causal={is_causal}")
        print(f"  -> grid={grid}, Q_BLOCK={Q_BLOCK}, K_BLOCK={K_BLOCK}")

        _flash_attn_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk, D,
            scale,
            is_causal,
            Q_BLOCK=Q_BLOCK,
            K_BLOCK=K_BLOCK,
        )
        print(f"[FlashAttentionTriton.forward] Kernel 执行完毕.")

        ctx.save_for_backward(Q, K, V, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        """
        反向传播 (进阶，留作练习!)

        提示: 需要重新计算 S_ij 和 P_ij (Recomputation)，
              然后计算 dQ, dK, dV。
        """
        raise NotImplementedError("Backward not implemented yet")


# ============================================================
# 第5步: 验证正确性
# ============================================================
def test_flash_attention():
    """对比 Triton FlashAttention 和 PyTorch scaled_dot_product_attention"""
    torch.manual_seed(42)

    configs = [
        (4, 128, 64),    # (batch, seq_len, head_dim)
        (2, 256, 128),
        (1, 512, 64),
    ]

    print("=== FlashAttention Forward 正确性验证 ===\n")

    for B, N, D in configs:
        print(f"\n[test_flash_attention] 测试配置: B={B}, N={N}, D={D}")
        Q = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, N, D, device="cuda", dtype=torch.float16)

        # PyTorch 参考实现
        o_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

        # Triton 实现
        o_triton = FlashAttentionTriton.apply(Q, K, V, is_causal=False)

        # 对比
        diff = (o_torch.float() - o_triton.float()).abs().max().item()
        rel_err = diff / o_torch.abs().float().max().item()
        status = "✅" if diff < 0.5 else "⚠️"
        print(f"  ({B}, {N}, {D}): max_diff={diff:.4e}, rel_err={rel_err:.2e} {status}")

    print()


def test_flash_attention_causal():
    """测试 Causal Masking"""
    torch.manual_seed(123)

    B, N, D = 2, 128, 64
    print(f"\n[test_flash_attention_causal] 测试配置: B={B}, N={N}, D={D}")
    Q = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, N, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, N, D, device="cuda", dtype=torch.float16)

    # PyTorch 因果实现
    o_torch = torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=True
    )

    # Triton 因果实现
    o_triton = FlashAttentionTriton.apply(Q, K, V, is_causal=True)

    diff = (o_torch.float() - o_triton.float()).abs().max().item()
    print(f"  Causal ({B}, {N}, {D}): max_diff={diff:.4e}")
    if diff < 0.5:
        print("  ✅ Causal 正确")
    else:
        print("  ⚠️ Causal 有较大差异，需要检查")


# ============================================================
# 第6步: 练习 —— 理解代码
# ============================================================
"""
思考题 (建议在完成代码后再看答案):

Q1: 为什么 M_acc 初始化为 -inf 而不是 0?
A1: 因为 max(-inf, x) = x。第一轮的 M_new 一定取 S_block 的 max。
    如果初始化为 0，第一轮可能会错误地用 0 和 S_block 取 max。

Q2: Online Softmax 中为什么要用 M_block 而不是 M_new 来计算 P_ij?
A2: 因为 P_ij = exp(S - M_block) 确保第一遍的数值稳定。
    M_new 是两块的合并最大值，但 P_ij 只是当前块的 softmax，
    应该用当前块的 max 来偏移。

Q3: 为什么 grid 是 (Nq//Q_BLOCK, B) 而不是 (Nq//Q_BLOCK, B, num_heads)?
A3: 当前实现不支持多 head。每个 kernel 只处理一个 head。
    要支持多 head，grid 需要是 (num_heads * Nq//Q_BLOCK, B)。

Q4: tl.make_block_ptr 的 order=(1,0) 是什么意思?
A4: 这表示 Triton 内部转置存储顺序。order=(1,0) 意味着:
    - 第 0 维 (行) 的 stride 对应顺序 (1)
    - 第 1 维 (列) 的 stride 对应逆序 (0)
    这让 Triton 能做更高效的内存合并访问 (coalesced access)。

Q5: 为什么 K_block_ptr 和 V_block_ptr 的 offsets 初始为 (0, 0)?
A5: 因为它们在 for 循环中用 advance() 逐步移动。
    每个 kernel 处理一个 Q block，但需要遍历所有 KV blocks。
"""


# ============================================================
# 第7步: 练习 —— 修改和实验
# ============================================================
"""
尝试修改以下参数，观察对正确性和性能的影响:

1. 改变 Q_BLOCK 和 K_BLOCK 的大小 (16, 32, 64, 128, 256)
2. 改变序列长度 N (32, 64, 128, 256, 512, 1024)
3. 改变 head dimension D (32, 64, 128)
4. 关闭 causal mask 再打开

记录你的发现!
"""


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        test_flash_attention()
        test_flash_attention_causal()
        print("\n🎉 第五章 FlashAttention 完成!")
        print("\n下一步: 阅读上面的思考题，尝试理解每个细节。")
        print("然后尝试实现 backward pass (第8章)!")
    else:
        print("需要 GPU 才能运行这些测试")
