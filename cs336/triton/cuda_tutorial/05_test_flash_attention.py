"""
CUDA 教程测试 — 第五章: FlashAttention Forward
================================================

对应 CUDA:   kernels/05_flash_attention.cu
对应 Triton: triton_tutorial/05_flash_attention.py
"""

import torch
import cuda_tutorial

print("=" * 60)
print("第五章: FlashAttention Forward (CUDA C++ 实现)")
print("=" * 60)

# ---- 正确性验证 ----
torch.manual_seed(42)

configs = [
    (4, 128, 64),     # (batch, seq_len, head_dim)
    (2, 256, 128),
    (1, 512, 64),
]

print("\n=== FlashAttention Forward 正确性验证 ===\n")

for B, N, D in configs:
    print(f"[flash_attn] 测试配置: B={B}, N={N}, D={D}")

    # next_power_of_2(D) 作为 blockDim.x
    block_dim = 1
    while block_dim < D:
        block_dim <<= 1
    print(f"  -> grid=({N}, {B}), block=({block_dim})")
    print(f"  -> K_BLOCK=32")
    print(f"  -> shared memory: ({block_dim} + 32) * 4 = {(block_dim + 32) * 4} bytes")

    Q = torch.randn(B, N, D, device="cuda")
    K = torch.randn(B, N, D, device="cuda")
    V = torch.randn(B, N, D, device="cuda")

    # PyTorch 参考实现 (标准 attention)
    scale = D ** -0.5
    S = torch.bmm(Q, K.transpose(1, 2)) * scale
    P = torch.softmax(S, dim=-1)
    o_torch = torch.bmm(P, V)

    # CUDA 实现
    o_cuda = cuda_tutorial.flash_attn(Q, K, V, is_causal=False)

    diff = (o_torch - o_cuda).abs().max().item()
    rel_err = diff / o_torch.abs().max().item()
    status = "✅" if diff < 0.01 else "⚠️"
    print(f"  max_diff={diff:.4e}, rel_err={rel_err:.2e} {status}\n")

# ---- Causal Masking ----
print("=== Causal Masking 测试 ===\n")

B, N, D = 2, 128, 64
print(f"[flash_attn_causal] 测试配置: B={B}, N={N}, D={D}")

Q = torch.randn(B, N, D, device="cuda")
K = torch.randn(B, N, D, device="cuda")
V = torch.randn(B, N, D, device="cuda")

# PyTorch 因果实现
scale = D ** -0.5
S = torch.bmm(Q, K.transpose(1, 2)) * scale
causal_mask = torch.tril(torch.ones(N, N, device="cuda")).bool()
S = S.masked_fill(~causal_mask, float("-inf"))
P = torch.softmax(S, dim=-1)
o_torch = torch.bmm(P, V)

# CUDA 因果实现
o_cuda = cuda_tutorial.flash_attn(Q, K, V, is_causal=True)

diff = (o_torch - o_cuda).abs().max().item()
print(f"  Causal max_diff={diff:.4e}")
if diff < 0.01:
    print("  ✅ Causal 正确")
else:
    print("  ⚠️ Causal 有较大差异")

print("\n🎉 第五章 CUDA 版完成!")
