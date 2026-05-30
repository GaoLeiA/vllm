"""
CUDA 教程测试 — 第三章: 行级聚合操作 (Softmax, LayerNorm)
=========================================================

对应 CUDA:   kernels/03_softmax.cu
对应 Triton: triton_tutorial/03_softmax.py
"""

import torch
import cuda_tutorial

print("=" * 60)
print("第三章: 行级聚合 — Softmax, LayerNorm (CUDA C++ 实现)")
print("=" * 60)

# ---- Softmax ----
M, N = 100, 256
x = torch.randn(M, N, device="cuda")

print(f"\n[softmax] 准备启动 Kernel:")
print(f"  -> M={M}, N={N}")
print(f"  -> grid=({M},), block_size=256")
print(f"  -> shared memory: 256 * 4 = 1024 bytes")

y_cuda = cuda_tutorial.softmax(x)
y_torch = torch.softmax(x, dim=-1)

diff = (y_cuda - y_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert torch.allclose(y_cuda, y_torch, atol=1e-5)
print("  ✅ 朴素 Softmax 正确")

# 测试极端值
x_extreme = torch.tensor([[1.0, -100.0, 100.0, -50.0]], device="cuda")
y_cuda_ext = cuda_tutorial.softmax(x_extreme)
y_torch_ext = torch.softmax(x_extreme, dim=-1)
diff = (y_cuda_ext - y_torch_ext).abs().max().item()
print(f"\n  极端值 max diff: {diff:.2e}")
assert torch.allclose(y_cuda_ext, y_torch_ext, atol=1e-4)
print("  ✅ 极端值 Softmax 正确")

# ---- LayerNorm ----
B, N = 4, 256
x = torch.randn(B, N, device="cuda")
gamma = torch.ones(N, device="cuda")
beta = torch.zeros(N, device="cuda")

print(f"\n[layernorm] 准备启动 Kernel:")
print(f"  -> M={B}, N={N}")
print(f"  -> grid=({B},), block_size=256")

y_cuda = cuda_tutorial.layernorm(x, gamma, beta, eps=1e-5)
y_torch = torch.nn.functional.layer_norm(x, (N,), weight=gamma, bias=beta)

diff = (y_cuda - y_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert torch.allclose(y_cuda, y_torch, atol=1e-4)
print("  ✅ LayerNorm 正确")

# 测试 gamma != 1, beta != 0
gamma2 = torch.randn(N, device="cuda")
beta2 = torch.randn(N, device="cuda")
y_cuda2 = cuda_tutorial.layernorm(x, gamma2, beta2, eps=1e-5)
y_torch2 = torch.nn.functional.layer_norm(x, (N,), weight=gamma2, bias=beta2)
diff2 = (y_cuda2 - y_torch2).abs().max().item()
print(f"\n  带参数的 LayerNorm max diff: {diff2:.2e}")
assert torch.allclose(y_cuda2, y_torch2, atol=1e-4)
print("  ✅ 带参数 LayerNorm 正确")

print("\n🎉 第三章 CUDA 版全部通过!")
