# 第七章：速查表与进阶资源

> **本章目的：** 提供一个完整的参考手册，涵盖前六章的核心知识。
> 建议打印或打开在另一个屏幕，写代码时随时参考。

---

## 1. Triton 编程模型速查

```
Grid (网格)
  ├── Block 0 (pid=0): 处理元素 [0, BLOCK_SIZE)
  ├── Block 1 (pid=1): 处理元素 [BLOCK_SIZE, 2*BLOCK_SIZE)
  └── Block N (pid=N): 处理元素 [N*BLOCK_SIZE, (N+1)*BLOCK_SIZE)

核心 API:
  tl.program_id(axis)     → 当前 block ID
  tl.arange(N)            → [0, 1, ..., N-1]
  tl.cdiv(a, b)           → ceil(a / b)
  tl.next_power_of_2(n)   → ≥n 的最小 2 的幂

内存操作:
  tl.load(ptr, mask=mask)  → 从全局内存读
  tl.store(ptr, val, mask) → 写回全局内存

Block Pointer:
  ptr = tl.make_block_ptr(base, shape, strides,
                          offsets, block_shape, order)
  ptr = ptr.advance((row_shift, col_shift))
  val = tl.load(ptr)
```

---

## 2. 算子模板速查

### 2.1 元素级操作

```python
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = your_formula(x)
    tl.store(y_ptr + offsets, y, mask=mask)

num_blocks = triton.cdiv(n, BLOCK_SIZE)
kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
```

### 2.2 行级聚合

```python
@triton.jit
def kernel(x_ptr, y_ptr, row_stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + row * row_stride + offsets,
                mask=offsets < N, other=...)
    # ... 聚合计算 ...
    y = ...
    tl.store(y_ptr + row * row_stride + offsets, y, mask=offsets < N)

kernel[(M,)](x, y, x.stride(0), N, BLOCK_SIZE=...)
```

### 2.3 矩阵乘法 Tiling

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(...)  # (BLOCK_M, BLOCK_K)
        b = tl.load(...)  # (BLOCK_K, BLOCK_N)
        acc += tl.dot(a, b)

    tl.store(C + ..., acc.to(C.type), mask=...)

kernel[(cdiv(M,BLOCK_M), cdiv(N,BLOCK_N))](...)
```

---

## 3. FlashAttention 公式速查

### 3.1 Forward

```
初始化: M=-inf, L=0, O=0

每轮 KV block:
  S = Q @ K^T / √D
  M_block = max(S, axis=1)
  M_new = max(M, M_block)
  P = exp(S - M_block)
  L_new = exp(M-M_new)·L + sum(P)·exp(M_block-M_new)
  O_new = exp(M-M_new)·O + P @ V

最终: Output = O / L
```

### 3.2 Backward

```
D = (dO × O).sum(dim=-1)

每对 (Q_block, KV_block):
  Recompute: S = Q_i @ K_j^T / √D
  Recompute: P = exp(S - m_i) / l_i
  dV_j += P^T @ dO_i
  dP = dO_i @ V_j^T
  dS = P × (dP - D)
  dQ_i += dS @ K_j × scale
  dK_j += dS^T @ Q_i × scale
```

---

## 4. tl 函数速查

```
tl.maximum(a, b)     → 逐元素 max
tl.exp(x)            → 逐元素 exp
tl.log(x)            → 逐元素 log
tl.sqrt(x)           → 逐元素 sqrt
tl.where(cond, a, b) → 条件选择
tl.dot(a, b)         → 矩阵乘法
tl.full(shape, val)  → 填充
tl.zeros(shape)      → 零
tl.arange(n)         → [0..n-1]
tl.cdiv(a, b)        → ceil(a/b)
tl.next_power_of_2(n)→ 2的幂
tl.sum(x, axis=0)    → 求和
tl.max(x, axis=0)    → 最大值
tl.tanh(a)           → ❌ 不存在! 用 (exp(2a)-1)/(exp(2a)+1)
```

---

## 5. 性能优化 Checklist

- [ ] BLOCK_SIZE 是 2 的幂 (128, 256, 512, 1024)
- [ ] 用 `tl.constexpr` 标记编译期常量
- [ ] 用 `tl.make_block_ptr` + `tl.load/store` (自动 coalescing)
- [ ] Grid 的 block 数量 ≥ 4 × SM 数 (H100: ≥ 528)
- [ ] 中间计算用 float32，最终写回时转 float16
- [ ] 用 `torch.profiler` 分析，不优化你以为的瓶颈
- [ ] 减少全局内存读写 (每个元素读一次，算完再写一次)

---

## 6. 调试 Checklist

- [ ] 用 `TRITON_INTERPRET=1` 在 CPU 上测试
- [ ] 用小块数据 (B=1, N=32, D=64) 开始
- [ ] 对比 PyTorch 的结果
- [ ] 检查数据类型 (中间 f32，最终 f16)
- [ ] 检查 mask 是否正确 (边界处理)
- [ ] 检查 stride 是否正确 (`tensor.stride()`)

---

## 7. 学习路径

```
Level 1 (1-2天):  完成 01-03 章 → 元素级 + 行级操作
Level 2 (3-5天):  完成 04-05 章 → MatMul + FlashAttention
Level 3 (1-2周):  完成 06 章 + 补全 backward + 做 Assignment 2
Level 4 (持续):   FlashAttention-2/3, vLLM kernel, Triton 源码
```

---

## 8. 参考资源

- **Triton:** https://github.com/triton-lang/triton
- **Tutorial:** https://triton-lang.org/main/getting-started/tutorials/
- **FA1 论文:** https://arxiv.org/abs/2205.14135
- **FA2 论文:** https://arxiv.org/abs/2307.08691
- **FA3 论文:** https://arxiv.org/abs/2407.08608
- **Horace He 博客:** https://horace.io/brrr_intro.html
- **CUDA MODE:** https://www.youtube.com/@CUDAMODE
- **GPU Puzzles:** https://github.com/srush/gpu-puzzles
- **CS336:** https://github.com/stanford-cs336/spring2025-lectures
- **diy-llm:** https://github.com/datawhalechina/diy-llm

---

> **恭喜你完成整个教程！** 从 Hello World 到 FlashAttention，你已经掌握了 Triton 的核心编程范式。
