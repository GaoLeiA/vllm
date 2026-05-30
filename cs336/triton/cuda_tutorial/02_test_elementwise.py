"""
CUDA 教程测试 — 第二章: 元素级操作 (Elementwise)
=================================================

对应 CUDA:   kernels/02_elementwise.cu
对应 Triton: triton_tutorial/02_elementwise.py
"""

import torch
import cuda_tutorial

print("=" * 60)
print("第二章: 元素级操作 — ReLU, Silu, GeLU, Add (CUDA C++ 实现)")
print("=" * 60)

N = 10000

# ---- ReLU ----
x = torch.randn(N, device="cuda")
y_cuda = cuda_tutorial.relu(x)
y_torch = torch.relu(x)

print(f"\n[relu] n={N}, block_size=1024, num_blocks={(N+1023)//1024}")
diff = (y_cuda - y_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert torch.allclose(y_cuda, y_torch, atol=1e-5)
print("  ✅ ReLU 正确")

# ---- Silu ----
y_cuda = cuda_tutorial.silu(x)
y_torch = torch.nn.functional.silu(x)

print(f"\n[silu] n={N}, block_size=1024, num_blocks={(N+1023)//1024}")
diff = (y_cuda - y_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert torch.allclose(y_cuda, y_torch, atol=1e-5)
print("  ✅ Silu 正确")

# ---- GeLU ----
y_cuda = cuda_tutorial.gelu(x)
y_torch = torch.nn.functional.gelu(x, approximate="tanh")

print(f"\n[gelu] n={N}, block_size=1024, num_blocks={(N+1023)//1024}")
diff = (y_cuda - y_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert torch.allclose(y_cuda, y_torch, atol=1e-5)
print("  ✅ GeLU 正确")

# ---- Add ----
x2 = torch.randn(N, device="cuda")
z_cuda = cuda_tutorial.add(x, x2)
z_torch = x + x2

print(f"\n[add] n={N}, block_size=1024, num_blocks={(N+1023)//1024}")
diff = (z_cuda - z_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert torch.allclose(z_cuda, z_torch, atol=1e-5)
print("  ✅ Element-wise Add 正确")

# ---- 性能对比 ----
import time

x_perf = torch.randn(1048576, device="cuda")
num_trials = 20

print(f"\n=== ReLU 性能对比 (1M 元素, {num_trials} 次平均) ===")

# PyTorch
torch.cuda.synchronize()
times = []
for _ in range(num_trials):
    start = time.time()
    _ = torch.relu(x_perf)
    torch.cuda.synchronize()
    times.append((time.time() - start) * 1000)
print(f"  PyTorch 原生:   {sum(times)/len(times):8.3f} ms")

# CUDA
torch.cuda.synchronize()
times = []
for _ in range(num_trials):
    start = time.time()
    _ = cuda_tutorial.relu(x_perf)
    torch.cuda.synchronize()
    times.append((time.time() - start) * 1000)
print(f"  CUDA C++:       {sum(times)/len(times):8.3f} ms")

print("\n🎉 第二章 CUDA 版全部通过!")
