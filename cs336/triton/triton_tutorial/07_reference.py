"""
Triton 从零开始教程 — 第七章: 速查表与进阶资源
==========================================

这是一个完整的速查表，涵盖前六章的所有核心概念。
建议打印出来或打开在另一个屏幕参考。
"""


# ============================================================
# 1. Triton 编程模型速查
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│              Triton 编程模型                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Grid (网格)                                            │
│  ├── Block 0: pid=0                                    │
│  │   ├── Thread 0: 处理元素 offset 0                   │
│  │   ├── Thread 1: 处理元素 offset 1                   │
│  │   └── ...                                           │
│  ├── Block 1: pid=1                                    │
│  │   ├── Thread 0: 处理元素 offset BLOCK_SIZE          │
│  │   └── ...                                           │
│  └── Block 2: pid=2                                    │
│      └── ...                                           │
│                                                         │
│  核心 API:                                              │
│    tl.program_id(axis)     → 当前 block 的 ID           │
│    tl.arange(N)            → [0, 1, ..., N-1]          │
│    tl.cdiv(a, b)           → ceil(a / b)               │
│    tl.next_power_of_2(n)   → 大于等于 n 的 2 的幂       │
│                                                         │
│  内存操作:                                              │
│    tl.load(ptr, mask=mask)  → 从全局内存读              │
│    tl.store(ptr, val, mask) → 写回全局内存              │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 2. 常用算子的 Triton 模板
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│  模板 1: 元素级操作 (Elementwise)                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  @triton.jit                                            │
│  def kernel(x_ptr, y_ptr, n, BLOCK_SIZE):               │
│      pid = tl.program_id(0)                             │
│      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)│
│      mask = offsets < n                                 │
│      x = tl.load(x_ptr + offsets, mask=mask)            │
│      y = ... # 你的计算                                  │
│      tl.store(y_ptr + offsets, y, mask=mask)            │
│                                                         │
│  # 启动:                                                │
│  num_blocks = triton.cdiv(n, BLOCK_SIZE)                │
│  kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=BLOCK_SIZE)  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  模板 2: 行级聚合 (Row-wise)                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  @triton.jit                                            │
│  def kernel(x_ptr, y_ptr, row_stride, N, BLOCK_SIZE):   │
│      row = tl.program_id(0)                             │
│      offsets = tl.arange(0, BLOCK_SIZE)                 │
│      x_row = tl.load(x_ptr + row * row_stride +         │
│                       offsets,                           │
│                       mask=offsets < N)                 │
│      # ... 聚合计算 (max, sum, etc.)                     │
│      y_row = ...                                        │
│      tl.store(y_ptr + row * row_stride + offsets,       │
│               y_row, mask=offsets < N)                  │
│                                                         │
│  # 启动:                                                │
│  kernel[(M,)](x, y, x.stride(0), N, BLOCK_SIZE=...)     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  模板 3: 矩阵乘法 (MatMul with Tiling)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  @triton.jit                                            │
│  def kernel(A, B, C, M, N, K, ...):                     │
│      pid_m = tl.program_id(0)                           │
│      pid_n = tl.program_id(1)                           │
│      offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) │
│      offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) │
│      acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=f32)      │
│      for k in range(0, K, BLOCK_K):                     │
│          a = tl.load(...)  # (BLOCK_M, BLOCK_K)         │
│          b = tl.load(...)  # (BLOCK_K, BLOCK_N)         │
│          acc += tl.dot(a, b)                            │
│      tl.store(C + ..., acc.to(C.type))                  │
│                                                         │
│  # 启动:                                                │
│  kernel[(cdiv(M,BM), cdiv(N,BN))](...)                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 3. FlashAttention 核心公式速查
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│  FlashAttention Forward                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  初始化: M=-inf, L=0, O=0                               │
│                                                         │
│  FOR each KV block j:                                   │
│    S = Q_i @ K_j^T / sqrt(D)                            │
│    (应用 causal mask)                                    │
│    M_block = max(S, axis=1)                             │
│    M_new = max(M_acc, M_block)                          │
│    P = exp(S - M_block)                                 │
│    L_new = exp(M_acc-M_new)*L + sum(P)*exp(M_block-M_new)│
│    O_new = exp(M_acc-M_new)*O + P @ V_j                 │
│    M = M_new, L = L_new, O = O_new                      │
│                                                         │
│  最终: Output = O / L                                   │
│        LogSumExp = M + log(L)                            │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  FlashAttention Backward                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  D = (dO * O).sum(dim=-1)  // 预计算                    │
│                                                         │
│  FOR each KV block j:                                   │
│    Recompute: S = Q_i @ K_j^T / sqrt(D)                │
│    Recompute: P = exp(S - m_i) / l_i                    │
│    dV_j += P^T @ dO_i                                   │
│    dP = dO_i @ V_j^T                                    │
│    dS = P * (dP - D_i)                                  │
│    dQ_i += dS @ K_j * scale                             │
│    dK_j += dS^T @ Q_i * scale                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 4. 常见 Triton 函数速查
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│  tl 常用函数                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  tl.max(x, axis=0)        → 沿 axis 取最大值            │
│  tl.sum(x, axis=0)        → 沿 axis 求和               │
│  tl.exp(x)                → 指数                         │
│  tl.log(x)                → 对数                         │
│  tl.sqrt(x)               → 平方根                       │
│  tl.where(cond, a, b)     → 条件选择                      │
│  tl.maximum(a, b)         → 逐元素取 max                 │
│  tl.dot(a, b)             → 矩阵乘法                      │
│  tl.full(shape, val)      → 填充张量                      │
│  tl.zeros(shape)          → 零张量                       │
│  tl.arange(n)             → [0, 1, ..., n-1]            │
│  tl.cdiv(a, b)            → ceil(a/b)                   │
│  tl.next_power_of_2(n)    → 2 的幂                       │
│                                                         │
│  tl.make_block_ptr(base, shape, strides,               │
│                     offsets, block_shape, order)        │
│    → 构造高效的 block pointer                           │
│                                                         │
│  ptr.advance(offsets)     → 移动 block pointer          │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 5. 性能优化技巧
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│  Triton 性能优化 Checklist                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ✅ 使用 tl.constexpr 标记编译期常量                     │
│     → Triton 编译器可以做更多优化                         │
│                                                         │
│  ✅ 让 BLOCK_SIZE 为 2 的幂                            │
│     → 硬件更高效，编译器更好优化                          │
│                                                         │
│  ✅ 使用 tl.make_block_ptr + tl.load/store             │
│     → 自动做 memory coalescing (合并访问)                │
│     → 比手动 offset 计算快 2-5x                         │
│                                                         │
│  ✅ Grid 的 block 数量 >= 4 × SM 数量                  │
│     → 保证足够的 occupancy                              │
│     → H100: 132 SM → 至少 528 blocks                   │
│                                                         │
│  ✅ 用 float32 做中间计算                               │
│     → 数值更稳定                                         │
│     → 只在最终写回时转回 fp16                            │
│                                                         │
│  ✅ 减少全局内存读写                                    │
│     → 一个元素读一次，算完再写一次                       │
│     → 用 shared memory (tl.dot 自动利用)                 │
│                                                         │
│  ✅ 用 torch.profiler 分析                            │
│     → 找出真正的瓶颈                                     │
│     → 不要优化你以为的瓶颈                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 6. 调试技巧
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│  Triton 调试 Checklist                                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. 先用 TRITON_INTERPRET=1 在 CPU 上运行                │
│     → 可以直接用 print() 调试                            │
│     → 可以看到每一步的结果                                │
│     → 缺点: 很慢!                                        │
│                                                         │
│  2. 用小块数据测试                                       │
│     → B=1, N=32, D=64 开始                            │
│     → 逐步增大                                          │
│     → 对比 PyTorch 的结果                               │
│                                                         │
│  3. 检查数据类型                                         │
│     → 很多错误来自 fp16 溢出                             │
│     → 中间变量用 f32                                    │
│                                                         │
│  4. 检查边界处理                                         │
│     → 序列长度不是 block_size 倍数时怎么办?              │
│     → mask 是否正确?                                    │
│                                                         │
│  5. 检查 stride                                         │
│     → 确保传入正确的 stride                             │
│     → torch.tensor.stride() 查看                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 7. 学习路径推荐
# ============================================================
"""
┌─────────────────────────────────────────────────────────┐
│  推荐学习路径                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Level 1 — 入门 (1-2 天)                                │
│  ├── 完成本教程的 01-03 章                               │
│  ├── 理解 Grid/Block/Thread 模型                        │
│  ├── 掌握 tl.load/tl.store 模式                         │
│  └── 能手写一个简单的激活函数 kernel                     │
│                                                         │
│  Level 2 — 进阶 (3-5 天)                                │
│  ├── 完成 04 章 MatMul (理解 Tiling)                    │
│  ├── 完成 05 章 FlashAttention Forward                  │
│  ├── 理解 Online Softmax 的数学                        │
│  └── 能在纸上画出 FlashAttention 的数据流图              │
│                                                         │
│  Level 3 — 高级 (1-2 周)                                │
│  ├── 完成 06 章 Backward Pass                           │
│  ├── 阅读 Triton 官方 tutorial:                          │
│  │   02-fused-softmax                                   │
│  │   03-matrix-multiplication                           │
│  │   06-fused-attention                                 │
│  ├── 实现 FlashAttention-2 (优化版)                     │
│  └── 尝试写 RMSNorm, GroupNorm 等 kernel                │
│                                                         │
│  Level 4 — 专家 (持续)                                  │
│  ├── 阅读 Triton 源码 (triton/language/core.py)         │
│  ├── 阅读 FlashAttention-2/3 论文                       │
│  ├── 参与 vLLM / Megatron-LM 的 kernel 优化              │
│  └── 研究新的 GPU 架构特性 (Hopper, Blackwell)          │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""


# ============================================================
# 8. 参考资源
# ============================================================
"""
官方资源:
  - Triton GitHub: https://github.com/triton-lang/triton
  - Triton Tutorial: https://triton-lang.org/main/getting-started/tutorials/
  - Triton Paper: https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf

FlashAttention:
  - FA1 论文: https://arxiv.org/abs/2205.14135
  - FA2 论文: https://arxiv.org/abs/2307.08691
  - FA3 论文: https://arxiv.org/abs/2407.08608

GPU 编程:
  - Horace He's Blog: https://horace.io/brrr_intro.html
  - CUDA MODE (YouTube): https://www.youtube.com/@CUDAMODE
  - GPU Puzzles: https://github.com/srush/gpu-puzzles

本课程配套:
  - Stanford CS336: https://github.com/stanford-cs336/spring2025-lectures
  - diy-llm (中文): https://github.com/datawhalechina/diy-llm
"""


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║        Triton 从零开始教程 — 全部完成!                    ║
║                                                          ║
║  你已经掌握了:                                            ║
║  ✅ 元素级操作 (ReLU, GeLU, Silu)                       ║
║  ✅ 行级聚合 (Softmax, LayerNorm)                       ║
║  ✅ 矩阵乘法 Tiling                                      ║
║  ✅ FlashAttention Forward (完整 Triton Kernel)          ║
║  ✅ FlashAttention Backward (框架 + PyTorch 验证)         ║
║  ✅ 速查表与性能优化技巧                                  ║
║                                                          ║
║  下一步:                                                  ║
║  1. 回到你的项目目录，运行每章代码                         ║
║  2. 尝试补全 backward pass                              ║
║  3. 阅读 assignment2-systems 的 triton_attention.py     ║
║  4. 实现并完成 Assignment 2                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
