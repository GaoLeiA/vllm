"""
CUDA 教程测试 — 第四章: 矩阵乘法 (Tiled MatMul)
================================================

对应 CUDA:   kernels/04_matmul.cu
对应 Triton: triton_tutorial/04_matmul.py
"""

import torch
import cuda_tutorial
import time

print("=" * 60)
print("第四章: 矩阵乘法 — Tiled MatMul (CUDA C++ 实现)")
print("=" * 60)

# ---- 基本正确性 ----
M, K, N = 128, 128, 128
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")

TILE_SIZE = 32
print(f"\n[matmul] 准备启动 Kernel:")
print(f"  -> M={M}, N={N}, K={K}")
print(f"  -> TILE_SIZE={TILE_SIZE}")
print(f"  -> grid=({(N+TILE_SIZE-1)//TILE_SIZE}, {(M+TILE_SIZE-1)//TILE_SIZE})")
print(f"  -> block=({TILE_SIZE}, {TILE_SIZE}) = {TILE_SIZE*TILE_SIZE} 线程")

C_cuda = cuda_tutorial.matmul(A, B)
C_torch = A @ B

diff = (C_cuda - C_torch).abs().max().item()
print(f"  max diff: {diff:.2e}")
assert diff < 0.1, f"差异过大: {diff}"
print("  ✅ 基本 MatMul 正确")

# ---- 不同尺寸 ----
print("\n=== 不同尺寸测试 ===")
sizes = [(64, 64, 64), (128, 256, 128), (256, 512, 256)]
for M, K, N in sizes:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")

    C_cuda = cuda_tutorial.matmul(A, B)
    C_torch = A @ B

    diff = (C_cuda - C_torch).abs().max().item()
    rel_err = diff / C_torch.abs().max().item()
    print(f"  ({M:3d}, {K:3d}, {N:3d}): max_diff={diff:.4e}, rel_err={rel_err:.2e}")

# ---- 性能对比 ----
print("\n=== 性能对比 ===")
sizes = [(512, 512, 512), (1024, 1024, 1024)]
num_trials = 20

for M, K, N in sizes:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")

    # 预热
    _ = A @ B
    _ = cuda_tutorial.matmul(A, B)
    torch.cuda.synchronize()

    # PyTorch (cuBLAS)
    times = []
    for _ in range(num_trials):
        start = time.time()
        _ = A @ B
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    avg_pytorch = sum(times) / len(times)

    # CUDA (我们的 tiled 实现)
    times = []
    for _ in range(num_trials):
        start = time.time()
        _ = cuda_tutorial.matmul(A, B)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    avg_cuda = sum(times) / len(times)

    print(f"  ({M:4d}, {K:4d}, {N:4d}): "
          f"PyTorch(cuBLAS)={avg_pytorch:7.3f}ms, "
          f"CUDA(Tiled)={avg_cuda:7.3f}ms")

print("\n🎉 第四章 CUDA 版全部通过!")
