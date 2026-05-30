"""
Triton 从零开始教程 — 第一章：环境与基本概念
==========================================

本教程配套代码仓库: cs336/triton
建议运行环境: GPU + Python 3.10+ + PyTorch + Triton

"""

# ============================================================
# 第0步: 环境检查
# ============================================================

import torch
import triton
import triton.language as tl

print(f"PyTorch 版本: {torch.__version__}")
print(f"Triton 版本: {triton.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# 第1步: 理解 GPU 编程的核心抽象
# ============================================================
"""
GPU 编程有三层概念，从大到小:

  Grid (网格) → Thread Block (线程块) → Thread (线程)

类比:
  Grid = 整个工厂
  Block = 一条流水线上的工人小组（共享一个工作台）
  Thread = 单个工人

每个 Block 内的线程可以:
  - 共享一块高速内存 (Shared Memory, 相当于 SRAM)
  - 同步等待 (barrier)
  - 协作完成一个子任务

每个 Thread 只执行一段代码，但处理不同的数据索引。

Triton 的关键简化:
  - 你不需要管理 Thread，只需要管理 Block
  - Shared Memory 和 Memory Coalescing 由编译器自动处理
  - 你只需要告诉 Triton: "每个 Block 处理哪块数据"
"""

# ============================================================
# 第2步: 你的第一个 Triton Kernel — Hello World
# ============================================================
"""
最简单的 Triton 内核: 把输入加 1

核心组件:
  1. @triton.jit 装饰器 — 标记这是一个 GPU 内核
  2. tl.program_id() — 获取当前线程块的 ID
  3. tl.arange() — 生成偏移量，让一个 Block 内的多个线程并行工作
  4. tl.load() / tl.store() — 从全局内存读写数据
"""

@triton.jit
def hello_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    参数说明:
      x_ptr, y_ptr: 输入/输出张量的指针 (C-style pointer)
      n_elements: 元素总数
      BLOCK_SIZE: 编译期常量 (constexpr)，告诉 Triton 每个 block 处理多少元素

    注意: BLOCK_SIZE 必须是 2 的幂，Triton 用它来生成高效的 mask
    """
    # 第1步: 当前是第几个 block?
    pid = tl.program_id(axis=0)

    # 第2步: 这个 block 从哪个位置开始处理?
    block_start = pid * BLOCK_SIZE

    # 第3步: 生成这个 block 内所有线程的偏移量
    # 例如 BLOCK_SIZE=1024 时: [0, 1, 2, ..., 1023]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 第4步: 创建 mask — 处理边界情况 (n_elements 可能不是 BLOCK_SIZE 的倍数)
    mask = offsets < n_elements

    # 第5步: 从全局内存读取数据 (带 mask，超出的位置不读)
    x = tl.load(x_ptr + offsets, mask=mask)

    # 第6步: 计算 — 每个线程处理一个元素
    y = x + 1

    # 第7步: 写回全局内存 (带 mask)
    tl.store(y_ptr + offsets, y, mask=mask)


# ============================================================
# 第3步: 启动 Kernel — Grid 配置
# ============================================================
def hello_wrapper(x: torch.Tensor) -> torch.Tensor:
    """包装函数: 把 Python 调用连接到 Triton Kernel"""
    assert x.is_cuda, "输入必须在 GPU 上"
    assert x.is_contiguous(), "输入必须是连续的"

    n = x.numel()
    y = torch.empty_like(x)

    # ---- Grid 配置 ----
    block_size = 1024  # 每个 block 处理 1024 个元素
    num_blocks = triton.cdiv(n, block_size)  # triton.cdiv = ceil(n / block_size)
    # 例如 n=5000, block_size=1024 → num_blocks=5

    print(f"\n[hello_wrapper] 准备启动 Kernel:")
    print(f"  -> 总元素数 (n): {n}")
    print(f"  -> 每个 Block 处理的元素数 (block_size): {block_size}")
    print(f"  -> 分配的 Block 数量 (num_blocks): {num_blocks}")

    # ---- 启动 Kernel ----
    # 语法: kernel_name[(grid_config)](参数列表)
    # grid_config 是一个 tuple，对应 kernel 中 program_id 的 axis
    hello_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)
    print(f"[hello_wrapper] Kernel 执行完毕.\n")

    return y


# ============================================================
# 第4步: 验证正确性
# ============================================================
if torch.cuda.is_available():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
    y = hello_wrapper(x)
    print(f"输入:  {x.cpu().tolist()}")
    print(f"输出:  {y.cpu().tolist()}")
    print(f"期望:  [2.0, 3.0, 4.0, 5.0, 6.0]")
    print(f"正确:  {torch.allclose(y.cpu(), x.cpu() + 1)}")

    # 测试大张量 + 非对齐边界
    x_big = torch.randn(1000000, device="cuda")
    y_big = hello_wrapper(x_big)
    assert torch.allclose(y_big.cpu(), x_big.cpu() + 1)
    print("大张量测试通过!")

    # 测试单元素
    x_single = torch.tensor([42.0], device="cuda")
    y_single = hello_wrapper(x_single)
    assert y_single.item() == 43.0
    print("单元素测试通过!")
