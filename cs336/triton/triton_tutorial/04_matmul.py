"""
Triton 从零开始教程 — 第四章: 矩阵乘法 (Matrix Multiplication)
=============================================================

本章目标: 掌握 Triton 最重要的编程模式 —— Tiling (分块) + Shared Memory。

矩阵乘法是深度学习中最核心的操作 (Attention, Linear layers 都依赖它)。
朴素实现的内存访问: M×K×N 次读, M×N 次写 —— 效率极低!

Tiled Matmul 的核心思想:
  把大矩阵切成小方块，每次只加载一小块到 SRAM，
  算完再加载下一块。这样每个元素只读一次 SRAM！

内存访问对比 (C = A @ B, 其中 A:(M,K), B:(K,N), C:(M,N)):
  朴素:  每算一个 C[i,j] 都要读 A[i,:] 和 B[:,j] → M×N×K 次读
  Tiled: 每个 tile 读一次 → M×K + K×N 次读  (降低 K 倍!)
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 练习 1: 最简单版 Matrix Mul — 不需要 Shared Memory
# ============================================================
"""
对于小矩阵 (K 很小)，可以直接在寄存器中完成计算。
这帮你理解 Triton 的矩阵乘法 API。

tl.dot(A, B): 矩阵乘法
  - A 形状: (num_tokens, K)
  - B 形状: (K, num_tokens)
  - 结果: (num_tokens, num_tokens)

注意: tl.dot 是 block matrix multiply，不是 element-wise
"""


@triton.jit
def matmul_simple_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    最简单的 matmul kernel (无 tiled, 无 shared memory)

    Grid: (num_m_blocks, num_n_blocks)
    每个 thread block 处理 C 的一个子块
    """
    # 确定当前 block 处理 C 的哪个子块
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # 子块的起始行/列
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 创建 mask
    mask_m = offset_m < M
    mask_n = offset_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # 初始化累加器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 循环遍历 K 维度
    for k in range(0, K, BLOCK_SIZE_K):
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offset_k < K

        # 加载 A 的子块 (M_block, K)
        a_ptrs = A + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载 B 的子块 (K, N_block)
        b_ptrs = B + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘法 (注意: tl.dot 要求内部维度对齐)
        acc += tl.dot(a, b)

    # 转回 float16 (如果输入是 fp16)
    acc = acc.to(C.element_type)

    # 写回
    c_ptrs = C + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask)


def matmul_simple_wrapper(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """包装函数"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Grid 配置
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    num_m_blocks = triton.cdiv(M, BLOCK_SIZE_M)
    num_n_blocks = triton.cdiv(N, BLOCK_SIZE_N)

    matmul_simple_kernel[(num_m_blocks, num_n_blocks)](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return C


# 验证
def test_matmul_simple():
    M, K, N = 128, 128, 128
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    C_torch = A @ B
    C_triton = matmul_simple_wrapper(A, B)

    # 注意: 我们的实现没有做数值优化，精度会稍差
    diff = (C_torch - C_triton).abs().max().item()
    print(f"  最大差异: {diff:.2e}")
    assert diff < 1.0, f"差异过大: {diff}"
    print("✅ 简单版 MatMul 基本正确")


# ============================================================
# 练习 2: 进阶版 —— 理解 tl.dot 的 tile 语义
# ============================================================
"""
上面的实现虽然能用，但效率不高。让我们深入理解为什么。

关键概念:
  tl.dot(A_tile, B_tile) 实际上做了:
    result = A_tile @ B_tile

  但 A_tile 和 B_tile 都在全局内存中。每次循环都要从 HBM (显存) 重新加载。
  这就是我们需要 Shared Memory + Tiling 的原因。

Triton 的 tl.dot 支持不同的 metadata 来优化:
  - 自动选择最优的实现 (SIMT, Tensor Core, etc.)
  - 取决于张量的数据类型和形状
"""


def test_matmul_sizes():
    """测试不同尺寸的矩阵乘法"""
    sizes = [(64, 64, 64), (128, 256, 128), (256, 512, 256)]
    for M, K, N in sizes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        C_torch = A @ B
        C_triton = matmul_simple_wrapper(A, B)

        diff = (C_torch - C_triton).abs().max().item()
        rel_err = diff / C_torch.abs().max().item()
        print(f"  ({M:3d}, {K:3d}, {N:3d}): max_diff={diff:.4e}, rel_err={rel_err:.2e}")


# ============================================================
# 练习 3: 性能对比
# ============================================================
def benchmark_matmul():
    """对比 Triton 简单版和 PyTorch 原生 matmul"""
    import time

    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    num_trials = 20

    for M, K, N in sizes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # 预热
        C1 = A @ B
        C2 = matmul_simple_wrapper(A, B)
        torch.cuda.synchronize()

        # PyTorch 计时
        times_pytorch = []
        for _ in range(num_trials):
            start = time.time()
            C1 = A @ B
            torch.cuda.synchronize()
            times_pytorch.append((time.time() - start) * 1000)

        # Triton 计时
        times_triton = []
        for _ in range(num_trials):
            start = time.time()
            C2 = matmul_simple_wrapper(A, B)
            torch.cuda.synchronize()
            times_triton.append((time.time() - start) * 1000)

        avg_pytorch = sum(times_pytorch) / len(times_pytorch)
        avg_triton = sum(times_triton) / len(times_triton)

        print(f"  ({M:4d}, {K:4d}, {N:4d}): "
              f"PyTorch={avg_pytorch:7.2f}ms, Triton={avg_triton:7.2f}ms, "
              f"ratio={avg_pytorch/avg_triton:.2f}x")


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("=== 简单版 MatMul ===")
        test_matmul_simple()
        print("\n=== 不同尺寸 ===")
        test_matmul_sizes()
        print("\n=== 性能对比 ===")
        benchmark_matmul()
        print("\n🎉 第四章全部通过!")
    else:
        print("需要 GPU 才能运行这些测试")
