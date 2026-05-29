"""
Triton 从零开始教程 — 第三章: 行级聚合操作 (Row-wise Aggregation)
================================================================

本章目标: 理解 Triton 中"一个 Block 处理一行"的模式。

Softmax 是第一个需要你跨多个元素做聚合操作的算子。
它的难点在于:
  - 需要遍历整行来算 max 和 sum
  - 需要两遍扫描: 先找最大值, 再算 softmax

这正是 Triton 中 "shared memory" 和 "block reduction" 的用武之地。

关键概念:
  1. 每个 Block 处理矩阵的一行
  2. Block 内线程通过 tl.max(), tl.sum() 做聚合
  3. 用 shared memory 避免重复读全局内存
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 练习 1: 朴素版 Softmax
# ============================================================
"""
朴素实现: 每个 Block 加载一整行，做两遍扫描

模式:
  pid = 0 → 第 0 行
  pid = 1 → 第 1 行
  ...

核心步骤:
  1. 加载一整行到寄存器
  2. 减去 max (数值稳定)
  3. exp
  4. sum
  5. 除以 sum
"""


@triton.jit
def softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    """
    参数:
      x_ptr, y_ptr: 输入输出指针
      x_row_stride, y_row_stride: 每行的字节偏移 (row stride)
      num_cols: 列数 (每行的元素个数)
      BLOCK_SIZE: 必须是 2 的幂，且 >= num_cols
    """
    # 当前处理第几行
    row_idx = tl.program_id(axis=0)

    # 生成所有列的偏移量
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 计算当前行的起始指针
    row_start = x_ptr + row_idx * x_row_stride

    # 加载整行 (带 mask 处理 num_cols < BLOCK_SIZE 的情况)
    x_row = tl.load(row_start + col_offsets, mask=col_offsets < num_cols, other=float("-inf"))

    # 第1遍: 找最大值 (用于数值稳定)
    x_max = tl.max(x_row, axis=0)

    # 减去 max (每列都减去同一值，不影响 softmax 结果)
    x_row = x_row - x_max

    # 第2遍: exp, sum, normalize
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 写回
    y_row_start = y_ptr + row_idx * y_row_stride
    tl.store(y_row_start + col_offsets, y_row, mask=col_offsets < num_cols)


def softmax_wrapper(x: torch.Tensor) -> torch.Tensor:
    """
    包装函数: 处理 (M, N) 的矩阵

    注意: Triton 的 softmax 是对最后一维 (每个 row) 做 softmax
    """
    M, N = x.shape
    y = torch.empty_like(x)

    block_size = triton.next_power_of_2(N)  # 取 2 的幂
    num_blocks = M  # 每行一个 block

    softmax_kernel[(M,)](
        x, y,
        x.stride(0), y.stride(0),  # row strides
        N, BLOCK_SIZE=block_size
    )
    return y


# 验证
def test_softmax():
    x = torch.randn(100, 256, device="cuda")
    y_torch = torch.softmax(x, dim=-1)
    y_triton = softmax_wrapper(x)
    assert torch.allclose(y_torch, y_triton, atol=1e-5)
    print("✅ 朴素 Softmax 正确")

    # 测试极端值 (数值稳定性)
    x_extreme = torch.tensor([[1.0, -100.0, 100.0, -50.0]], device="cuda")
    y_torch_extreme = torch.softmax(x_extreme, dim=-1)
    y_triton_extreme = softmax_wrapper(x_extreme)
    assert torch.allclose(y_torch_extreme, y_triton_extreme, atol=1e-4)
    print("✅ 极端值 Softmax 正确")


# ============================================================
# 练习 2: 优化版 Softmax — 减少全局内存读写
# ============================================================
"""
朴素版的问题:
  - 每行读 num_cols 次全局内存 (max, exp, sum, normalize)
  - 每行写 num_cols 次全局内存

优化思路:
  - 一次加载整行到寄存器
  - 在寄存器中完成所有计算
  - 只写一次

实际上上面的朴素版已经是这样了！但我们可以进一步优化:
  使用 tl.reduce 和更少的中间变量
"""


@triton.jit
def softmax_optimized_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start = x_ptr + row_idx * x_row_stride
    x_row = tl.load(row_start + col_offsets, mask=col_offsets < num_cols, other=float("-inf"))

    # 关键优化: 用同一个变量存储中间结果，减少寄存器压力
    x_max = tl.max(x_row, axis=0)
    x_row = x_row - x_max
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    y_row_start = y_ptr + row_idx * y_row_stride
    tl.store(y_row_start + col_offsets, y_row, mask=col_offsets < num_cols)


# 验证优化版 (与朴素版等价)
def test_softmax_optimized():
    x = torch.randn(100, 256, device="cuda")
    y1 = softmax_wrapper(x)
    block_size = triton.next_power_of_2(256)
    y2 = torch.empty_like(x)
    softmax_optimized_kernel[(100,)](
        x, y2,
        x.stride(0), y2.stride(0),
        256, BLOCK_SIZE=block_size
    )
    assert torch.allclose(y1, y2, atol=1e-5)
    print("✅ 优化版 Softmax 与朴素版等价")


# ============================================================
# 练习 3: Layer Normalization — Softmax 的亲戚
# ============================================================
"""
LayerNorm(x) = (x - mean(x)) / std(x) * gamma + beta

这和 Softmax 的"整行聚合"模式完全一样!
"""


@triton.jit
def layernorm_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr, row_stride, num_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start = x_ptr + row_idx * row_stride
    x_row = tl.load(row_start + col_offsets, mask=col_offsets < num_cols, other=0.0)

    # 计算 mean 和 var
    mean = tl.sum(x_row, axis=0) / num_cols
    var = tl.sum((x_row - mean) * (x_row - mean), axis=0) / num_cols

    # normalize
    x_norm = (x_row - mean) / tl.sqrt(var + eps)

    # 加载 gamma 和 beta (这两个是 per-feature 的，每个元素不同)
    gamma = tl.load(gamma_ptr + col_offsets, mask=col_offsets < num_cols)
    beta = tl.load(beta_ptr + col_offsets, mask=col_offsets < num_cols)

    y_row = x_norm * gamma + beta

    y_row_start = y_ptr + row_idx * row_stride
    tl.store(y_row_start + col_offsets, y_row, mask=col_offsets < num_cols)


def layernorm_wrapper(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(N)
    layernorm_kernel[(M,)](
        x, y, gamma, beta,
        x.stride(0), N, eps, BLOCK_SIZE=block_size
    )
    return y


def test_layernorm():
    B, N = 4, 256
    x = torch.randn(B, N, device="cuda")
    gamma = torch.ones(N, device="cuda")
    beta = torch.zeros(N, device="cuda")

    y_torch = torch.nn.functional.layer_norm(x, (N,), weight=gamma, bias=beta)
    y_triton = layernorm_wrapper(x, gamma, beta)

    assert torch.allclose(y_torch, y_triton, atol=1e-4)
    print("✅ LayerNorm 正确")


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        test_softmax()
        test_softmax_optimized()
        test_layernorm()
        print("\n🎉 第三章全部通过!")
    else:
        print("需要 GPU 才能运行这些测试")
