"""
CUDA 教程测试 — 第一章: Hello Kernel (add 1)
=============================================

对应 CUDA:   kernels/01_hello.cu
对应 Triton: triton_tutorial/01_introduction.py
"""

import torch
import cuda_tutorial

print("=" * 60)
print("第一章: Hello Kernel — 向量加 1 (CUDA C++ 实现)")
print("=" * 60)

# ---- 测试 1: 小向量 ----
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
print(f"\n[hello] 准备启动 Kernel:")
print(f"  -> 总元素数: {x.numel()}")
print(f"  -> block_size: 1024, num_blocks: {(x.numel() + 1023) // 1024}")

y = cuda_tutorial.hello(x)

print(f"  输入:  {x.cpu().tolist()}")
print(f"  输出:  {y.cpu().tolist()}")
print(f"  期望:  [2.0, 3.0, 4.0, 5.0, 6.0]")
assert torch.allclose(y.cpu(), x.cpu() + 1)
print("  ✅ 小向量测试通过!")

# ---- 测试 2: 大张量 + 非对齐边界 ----
x_big = torch.randn(1000000, device="cuda")
num_blocks = (1000000 + 1023) // 1024
print(f"\n[hello] 大张量测试:")
print(f"  -> 总元素数: {x_big.numel()}")
print(f"  -> block_size: 1024, num_blocks: {num_blocks}")

y_big = cuda_tutorial.hello(x_big)
assert torch.allclose(y_big.cpu(), x_big.cpu() + 1)
print("  ✅ 大张量测试通过!")

# ---- 测试 3: 单元素 ----
x_single = torch.tensor([42.0], device="cuda")
y_single = cuda_tutorial.hello(x_single)
assert y_single.item() == 43.0
print("\n  ✅ 单元素测试通过!")

print("\n🎉 第一章 CUDA 版全部通过!")
