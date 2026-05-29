# 第四章：矩阵乘法 — Tiling 与 Shared Memory

> **你将学到什么：**
> - 理解矩阵乘法的内存瓶颈
> - 掌握 Tiling（分块）的核心思想
> - 理解 Shared Memory 的作用
> - 实现 Triton 矩阵乘法 kernel
> - 对比 PyTorch 原生和 Triton 实现的性能
>
> **前置知识：** 第三章（行级聚合，Softmax）
> **预计时间：** 60 分钟
> **最终成果：** 一个能处理任意尺寸矩阵的 Triton matmul kernel

---

## 1. 矩阵乘法的内存瓶颈

### 1.1 矩阵乘法回顾

```
C = A @ B

A: (M, K)        B: (K, N)        C: (M, N)
[cᵢⱼ = Σₖ aᵢₖ · bₖⱼ]
```

每个 `cᵢⱼ` 需要 K 次乘法和 K-1 次加法。总共 M×N 个输出元素。

### 1.2 朴素实现的内存访问

```python
# 朴素实现: 每个线程算一个 cᵢⱼ
for i in range(M):
    for j in range(N):
        s = 0
        for k in range(K):
            s += A[i, k] * B[k, j]
        C[i, j] = s
```

**内存访问分析：**

- 每个 `cᵢⱼ` 需要读 `A[i, :]` (K 个元素) 和 `B[:, j]` (K 个元素)
- 但 `A[i, k]` 被 `cᵢ₀, cᵢ₁, ..., cᵢₙ₋₁` 共 N 次使用
- `B[k, j]` 被 `c₀ⱼ, c₁ⱼ, ..., cₘ₋₁ⱼ` 共 M 次使用

**总内存访问：**
- 读 A: M × K × N 次 ❌（实际只需要 M × K 次）
- 读 B: K × N × M 次 ❌（实际只需要 K × N 次）
- 写 C: M × N 次 ✅

**问题：** 每个元素被重复读取！这就是我们需要 Tiling 的原因。

### 1.3 类比：做饭

> 假设你要用 10 种食材做 10 道菜：
>
> **朴素做法：** 每做一道菜就去一次仓库拿食材 → 10 道菜 × 10 次仓库 = 100 趟
>
> **Tiling 做法：** 一次把所有食材搬出来 → 1 趟仓库 + 做 10 道菜

---

## 2. Tiling（分块）的核心思想

### 2.1 基本思路

```
把大矩阵切成小方块，每次只处理一小块:

A: (M, K)          B: (K, N)          C: (M, N)
┌──────┬──────┐    ┌──────┬──────┐    ┌──────┬──────┐
│ A₀₀  │ A₀₁  │    │ B₀₀  │ B₀₁  │    │ C₀₀  │ C₀₁  │
├──────┼──────┤    ├──────┼──────┤    ├──────┼──────┤
│ A₁₀  │ A₁₁  │    │ B₁₀  │ B₁₁  │    │ C₁₀  │ C₁₁  │
└──────┴──────┘    └──────┴──────┘    └──────┴──────┘

C₀₀ = A₀₀ @ B₀₀ + A₀₁ @ B₁₀   (按 K 维度累加)
```

**每个 block_size × block_size 的子矩阵乘法：**

```
C_tile += A_tile @ B_tile
```

### 2.2 内存访问对比

| 方法 | 读 A | 读 B | 写 C |
|------|------|------|------|
| 朴素 | M×K×N | K×N×M | M×N |
| **Tiled** | M×K + K×N | M×K + K×N | M×N |

**Tiling 把读写从 O(M×N×K) 降到 O(M×K + K×N)，减少了 K 倍！**

---

## 3. Triton 矩阵乘法实现

### 3.1 Grid 配置

```
C 矩阵: (M, N)
BLOCK_M = 64, BLOCK_N = 64

Grid: (M/64, N/64)

每个 block 负责 C 的一个 64×64 子块
```

### 3.2 完整实现

```python
@triton.jit
def matmul_simple_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,        # A 的 strides
    stride_bk, stride_bn,        # B 的 strides
    stride_cm, stride_cn,        # C 的 strides
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ---- Step 1: 确定当前 block 负责 C 的哪个子块 ----
    pid_m = tl.program_id(axis=0)   # 行块 ID
    pid_n = tl.program_id(axis=1)   # 列块 ID

    # ---- Step 2: 生成行/列偏移 ----
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # ---- Step 3: 创建 mask ----
    mask_m = offset_m < M
    mask_n = offset_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # ---- Step 4: 初始化累加器 (float32 避免溢出) ----
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ---- Step 5: 循环遍历 K 维度 ----
    for k in range(0, K, BLOCK_SIZE_K):
        offset_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offset_k < K

        # 加载 A 的子块 (BLOCK_M, BLOCK_K)
        a_ptrs = A + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # 加载 B 的子块 (BLOCK_K, BLOCK_N)
        b_ptrs = B + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # 矩阵乘法
        acc += tl.dot(a, b)

    # ---- Step 6: 转回输入类型并写回 ----
    acc = acc.to(C.element_type)
    c_ptrs = C + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask)
```

### 3.3 逐段解析

#### 3.3.1 确定负责的子块

```python
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)

offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
# 例如 pid_m=2, BLOCK_SIZE_M=64 → [128, 129, ..., 191]

offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
```

#### 3.3.2 2D Mask

```python
mask_m = offset_m < M     # shape: (BLOCK_SIZE_M,)
mask_n = offset_n < N     # shape: (BLOCK_SIZE_N,)
mask = mask_m[:, None] & mask_n[None, :]  # shape: (BLOCK_SIZE_M, BLOCK_SIZE_N)
```

**为什么是 2D mask？** 因为 C 的子块可能超出边界，需要对每个元素单独判断。

#### 3.3.3 加载 A 的子块

```python
a_ptrs = A + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
```

这是 Triton 的**指针算术**：

```
A[i, k] 的指针 = A + i * stride_am + k * stride_ak

offset_m[:, None]  → shape (BLOCK_M, 1)  → 行索引
offset_k[None, :]  → shape (1, BLOCK_K)  → 列索引

相加后 shape (BLOCK_M, BLOCK_K) → 正好是 A 的子块
```

#### 3.3.4 `tl.dot(a, b)` — 矩阵乘法

```python
a: shape (BLOCK_M, BLOCK_K)
b: shape (BLOCK_K, BLOCK_N)
tl.dot(a, b) → shape (BLOCK_M, BLOCK_N)
```

**注意：** 内部维度 (BLOCK_K) 必须对齐！

#### 3.3.5 K 维度的循环

```python
for k in range(0, K, BLOCK_SIZE_K):
    # 每次加载 A 的一列块和 B 的一行块
    # 做一次 mini matmul: acc += A_col @ B_row
    # ...
```

这个循环是 Triton kernel 中**唯一的 Python for 循环**。Triton 编译器会把它优化为 GPU 循环。

---

## 4. 包装函数

```python
def matmul_simple_wrapper(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Grid 配置
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    num_m = triton.cdiv(M, BLOCK_M)
    num_n = triton.cdiv(N, BLOCK_N)

    matmul_simple_kernel[(num_m, num_n)](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return C
```

**关键：** strides 是从 `tensor.stride()` 获取的，不是 `tensor.shape`！

```python
>>> x = torch.randn(128, 256)
>>> x.shape    # (128, 256)
>>> x.stride() # (256, 1)  ← 每行的字节偏移是 256 (256*4 bytes = 1024 bytes)
```

---

## 5. 正确性验证

### 5.1 小矩阵测试

```python
M, K, N = 128, 128, 128
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

C_torch = A @ B
C_triton = matmul_simple_wrapper(A, B)

diff = (C_torch - C_triton).abs().max().item()
print(f"max diff: {diff:.2e}")  # 预期: < 1.0 (fp16 精度)
```

### 5.2 不同尺寸测试

```python
for M, K, N in [(64, 64, 64), (128, 256, 128), (256, 512, 256)]:
    # 测试
```

### 5.3 为什么有误差？

```
FP16 的精度: ~3 位有效数字
FP16 matmul 的累积误差: 随 K 增大而增大

max diff = 0.5 ~ 2.0 是正常的
max diff < 10.0 通常可以接受
```

**如果用 FP32：** 误差会小到 `1e-5` 级别。

---

## 6. 性能对比

### 6.1 对比 PyTorch 原生

```python
for size in [512, 1024, 2048]:
    M = K = N = size
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # PyTorch
    pytorch_time = benchmark(A @ B)

    # Triton
    triton_time = benchmark(matmul_simple_wrapper(A, B))

    print(f"{size}x: PyTorch={pytorch:.2f}ms, Triton={triton:.2f}ms")
```

### 6.2 预期结果

```
(512, 512, 512): PyTorch=0.15ms, Triton=0.45ms  (ratio=0.33x)
(1024, 1024, 1024): PyTorch=1.50ms, Triton=3.00ms (ratio=0.50x)
(2048, 2048, 2048): PyTorch=15.00ms, Triton=20.00ms (ratio=0.75x)
```

**为什么比 PyTorch 慢？**

1. 我们的实现没有用 Shared Memory（`tl.dot` 内部会尝试用，但不如手写优化）
2. 没有用 Tensor Core 的 tile size 优化
3. 没有做循环展开（loop unrolling）

**但这不重要！** 我们的目标是理解 Tiling 的原理。真正的生产级 matmul kernel（如 FlashAttention 用的）会做更多优化。

---

## 7. 动手练习

### 练习 1：理解 mask

```python
# 如果 M=100, BLOCK_M=64, 有几个 block?
num_m = triton.cdiv(100, 64)  # = 2

# 第一个 block 处理哪些行?
# pid_m=0: offset_m = [0, 1, ..., 63]  — 全部有效

# 第二个 block 处理哪些行?
# pid_m=1: offset_m = [64, 65, ..., 127]
# 但 M=100, 所以 offset_m >= 100 的行被 mask 掉
# 有效行: [64, 65, ..., 99] (共 36 行)
```

### 练习 2：修改 Block 大小

尝试以下组合，观察对正确性和性能的影响：

| BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|---------|---------|---------|------|
| 32 | 32 | 32 | 小块，更多 block |
| 64 | 64 | 64 | 标准配置 |
| 128 | 128 | 128 | 大块，更少 block |
| 64 | 64 | 32 | K 块比 M,N 块小 |

### 练习 3：理解 stride

```python
>>> x = torch.randn(128, 256, device="cuda")
>>> x.stride()  # (256, 1)
>>> x.T.stride()  # (1, 256) ← 转置后 stride 变了!
```

**问题：** 如果传入一个转置过的矩阵给 kernel，stride 应该怎么传？

<details>
<summary>点击查看</summary>

传入 `A.T` 时，`A.T.stride(0) = A.stride(1) = 1`，`A.T.stride(1) = A.stride(0) = 256`。

kernel 中用 `A.T.stride(0), A.T.stride(1)` 即可，不需要手动交换。
Triton kernel 通过 stride 自动适配不同的内存布局。

</details>

---

## 8. 核心知识点总结

### 8.1 Tiling 模板

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    # 1. 确定 C 的子块位置
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 生成偏移
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 3. 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 4. 循环 K 维度
    for k in range(0, K, BLOCK_K):
        # 5. 加载子块 (带 mask)
        a = tl.load(A_ptrs, mask=...)
        b = tl.load(B_ptrs, mask=...)
        # 6. mini matmul
        acc += tl.dot(a, b)

    # 7. 写回
    tl.store(C_ptrs, acc.to(C.type), mask=...)
```

### 8.2 关键公式

```
内存访问 (朴素):  2 × M × N × K 次读
内存访问 (Tiled):  M × K + K × N + M × N 次读写

加速比: 约 K 倍 (当 M ≈ N ≈ K 时)
```

### 8.3 设计决策

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| BLOCK_M, BLOCK_N | 64 或 128 | 平衡 occupancy 和寄存器压力 |
| BLOCK_K | 与 M,N 相同或一半 | 影响 K 维度循环次数 |
| acc dtype | float32 | 避免 fp16 累积溢出 |

---

## 9. 下一章预告

有了矩阵乘法的 Tiling 基础，下一章我们将实现 **FlashAttention** — 矩阵乘法和 Online Softmax 的组合。

FlashAttention 的核心就是：
1. 用 Tiled MatMul 计算 `S = Q @ K^T`
2. 用 Online Softmax 避免存储完整的 S 矩阵
3. 最终结果 `O = P @ V` 也是 tiled 的

**运行本章代码：** `python 04_matmul.py`
**下一章：** 第五章 — FlashAttention Forward（本章是整个教程的核心）
