"""
Triton 从零开始教程 — 第二章: 元素级操作 (Elementwise Operations)
================================================================

本章目标: 掌握 Triton 中最基本的内核编写模式 —— 元素级操作。
我们将实现三个激活函数: GeLU, ReLU, Silu (Swish)。

核心模式:
  1. 确定 grid (有多少个 block)
  2. 每个 block 生成 offset 序列
  3. 创建 mask 处理边界
  4. tl.load() → 计算 → tl.store()
"""

import torch
import triton
import triton.language as tl


# ============================================================
# 练习 1: ReLU — 最简单的非线性激活
# ============================================================
"""
ReLU(x) = max(0, x)

实现思路: 每个线程处理一个元素，如果 x < 0 则输出 0
"""


@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)  # ReLU = max(0, x)
    tl.store(y_ptr + offsets, y, mask=mask)


def relu_wrapper(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)
    y = torch.empty_like(x)
    print(f"\n[relu_wrapper] 准备启动 Kernel: n={n}, block_size={block_size}, num_blocks={num_blocks}")
    relu_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)
    print(f"[relu_wrapper] Kernel 执行完毕.")
    return y


# 验证
def test_relu():
    x = torch.randn(1000, device="cuda")
    y_torch = torch.relu(x)
    y_triton = relu_wrapper(x)
    assert torch.allclose(y_torch, y_triton, atol=1e-5)
    print("✅ ReLU 正确")


# ============================================================
# 练习 2: Silu (Swish) — 带乘法的激活函数
# ============================================================
"""
Silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

注意: 这里涉及两个元素的组合操作 (x 和 sigmoid(x))，
但仍然是逐元素计算，所以一个线程处理一个元素的模式不变。
"""


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid_x
    tl.store(y_ptr + offsets, y, mask=mask)


def silu_wrapper(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)
    y = torch.empty_like(x)
    print(f"\n[silu_wrapper] 准备启动 Kernel: n={n}, block_size={block_size}, num_blocks={num_blocks}")
    silu_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)
    print(f"[silu_wrapper] Kernel 执行完毕.")
    return y


# 验证
def test_silu():
    x = torch.randn(1000, device="cuda")
    y_torch = torch.nn.functional.silu(x)
    y_triton = silu_wrapper(x)
    assert torch.allclose(y_torch, y_triton, atol=1e-5)
    print("✅ Silu 正确")


# ============================================================
# 练习 3: GeLU — 带复杂公式的激活函数
# ============================================================
"""
GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

注意: Triton 没有 tl.tanh，需要手动展开:
  tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
"""


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # GeLU 近似公式 (tanh 版本，与 PyTorch 一致)
    # sqrt(2/pi) ≈ 0.79788456
    a = 0.79788456 * (x + 0.044715 * x * x * x)

    # 手动展开 tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    exp_2a = tl.exp(2 * a)
    tanh_a = (exp_2a - 1.0) / (exp_2a + 1.0)

    y = 0.5 * x * (1.0 + tanh_a)
    tl.store(y_ptr + offsets, y, mask=mask)


def gelu_wrapper(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)
    y = torch.empty_like(x)
    print(f"\n[gelu_wrapper] 准备启动 Kernel: n={n}, block_size={block_size}, num_blocks={num_blocks}")
    gelu_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)
    print(f"[gelu_wrapper] Kernel 执行完毕.")
    return y


# 验证
def test_gelu():
    x = torch.randn(1000, device="cuda")
    y_torch = torch.nn.functional.gelu(x, approximate="tanh")
    y_triton = gelu_wrapper(x)
    assert torch.allclose(y_torch, y_triton, atol=1e-5)
    print("✅ GeLU 正确")


# ============================================================
# 练习 4: 对比三种实现方式
# ============================================================
"""
同一操作 (ReLU)，三种写法:
  1. 手写 Python 循环 (最慢，纯 CPU)
  2. PyTorch 原生 (中等速度，C++ 后端)
  3. Triton (最快，GPU kernel)

这个对比能让你直观感受到 Triton 的价值。
"""


def manual_relu(x: torch.Tensor) -> torch.Tensor:
    """纯 Python 实现，仅用于对比"""
    result = torch.empty_like(x)
    for i in range(x.numel()):
        result[i] = max(0.0, x[i])
    return result


def benchmark_function(func, x, name, num_trials=10):
    """简单计时"""
    # 预热
    func(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    import time
    times = []
    for _ in range(num_trials):
        start = time.time()
        func(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # ms

    print(f"  {name:20s}: {sum(times)/len(times):8.2f} ms (平均 {num_trials} 次)")


def benchmark_comparison():
    """对比三种 ReLU 实现"""
    x = torch.randn(1048576, device="cuda")  # 1M 元素

    print("\n=== ReLU 性能对比 (1M 元素) ===")
    benchmark_function(manual_relu, x, "手写 Python")
    benchmark_function(torch.relu, x, "PyTorch 原生")
    benchmark_function(relu_wrapper, x, "Triton")


# ============================================================
# 练习 5: 多参数操作
# ============================================================
"""
现在试试有两个输入的操作: element-wise add, mul, 等
"""


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(x_ptr + offsets, mask=mask)
    b = tl.load(y_ptr + offsets, mask=mask)
    c = a + b
    tl.store(z_ptr + offsets, c, mask=mask)


def add_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)
    z = torch.empty_like(x)
    print(f"\n[add_wrapper] 准备启动 Kernel: n={n}, block_size={block_size}, num_blocks={num_blocks}")
    add_kernel[(num_blocks,)](x, y, z, n, BLOCK_SIZE=block_size)
    print(f"[add_wrapper] Kernel 执行完毕.")
    return z


def test_add():
    x = torch.randn(1000, device="cuda")
    y = torch.randn(1000, device="cuda")
    z_torch = x + y
    z_triton = add_wrapper(x, y)
    assert torch.allclose(z_torch, z_triton, atol=1e-5)
    print("✅ Element-wise Add 正确")


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    if torch.cuda.is_available():
        test_relu()
        test_silu()
        test_gelu()
        test_add()
        benchmark_comparison()
        print("\n🎉 第二章全部通过!")
    else:
        print("需要 GPU 才能运行这些测试")
