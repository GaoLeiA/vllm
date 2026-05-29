# 第二章：元素级操作 — ReLU, GeLU, Silu

> **你将学到什么：**
> - 实现三种常见激活函数的 Triton kernel
> - 理解 `tl.maximum`, `tl.exp`, `tl.where` 等常用函数
> - 手动展开 `tanh` 公式（因为 Triton 没有 `tl.tanh`）
> - 对比 PyTorch 原生实现和 Triton 实现的性能
>
> **前置知识：** 第一章（Hello World 模板）
> **预计时间：** 45 分钟
> **最终成果：** 3 个激活函数 kernel + 性能对比程序

---

## 1. 元素级操作的本质

元素级操作（Elementwise Operation）是最简单的 GPU 并行任务：

```
输入: [x₀, x₁, x₂, ..., xₙ₋₁]    每个元素独立
输出: [f(x₀), f(x₁), f(x₂), ..., f(xₙ₋₁)]  每个结果只依赖对应输入
```

**每个线程处理一个元素，线程之间没有任何依赖。** 这就是元素级操作最简单的原因。

### 1.1 本章要实现的三个激活函数

| 激活函数 | 公式 | 用途 |
|---------|------|------|
| **ReLU** | `max(0, x)` | 最常用的非线性激活 |
| **Silu** | `x · σ(x) = x / (1 + e⁻ˣ)` | Swish 函数，GPT-2/3 使用 |
| **GeLU** | `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))` | Transformer 标配 |

---

## 2. ReLU — 最简单的非线性

### 2.1 公式

```
ReLU(x) = max(0, x)
         = { x,  if x > 0
         { 0,  if x ≤ 0
```

### 2.2 Triton 实现

```python
@triton.jit
def relu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)  # ← 这就是 max(0, x)
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 2.3 逐行理解

```
tl.maximum(x, 0.0)
```

这就是 `max(0, x)` 的 Triton 写法。它自动向量化 — 对 `offsets` 中的每个元素都执行 `max`。

**注意：** 不需要写 `if` 判断。GPU 的 SIMD 架构会自动处理分支。

---

## 3. Silu (Swish) — 带除法的激活函数

### 3.1 公式

```
Silu(x) = x · σ(x)
        = x / (1 + e⁻ˣ)
```

### 3.2 Triton 实现

```python
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # σ(x) = 1 / (1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid_x

    tl.store(y_ptr + offsets, y, mask=mask)
```

### 3.3 理解

这个 kernel 展示了 Triton 的**向量化计算**：
- `tl.exp(-x)` — 对每个元素求 `exp(-x)`
- `1.0 / (1.0 + ...)` — 每个元素做除法
- `x * sigmoid_x` — 每个元素做乘法

**全部是向量化的，一行代码等于 N 个元素的计算。**

---

## 4. GeLU — 最复杂的元素级操作

### 4.1 公式

```
GeLU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

### 4.2 问题：Triton 没有 `tl.tanh`

Triton 没有提供 `tl.tanh` 函数，但我们可以手动展开：

```
tanh(a) = (eᵃ - e⁻ᵃ) / (eᵃ + e⁻ᵃ)
        = (e²ᵃ - 1) / (e²ᵃ + 1)
```

推导：
```
tanh(a) = sinh(a) / cosh(a)
        = (eᵃ - e⁻ᵃ) / (eᵃ + e⁻ᵃ)     ← 分子分母同乘 eᵃ
        = (e²ᵃ - 1) / (e²ᵃ + 1)        ← 这就是我们要的
```

### 4.3 Triton 实现

```python
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # 常量
    # √(2/π) ≈ 0.79788456
    a = 0.79788456 * (x + 0.044715 * x * x * x)

    # 展开 tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    exp_2a = tl.exp(2 * a)
    tanh_a = (exp_2a - 1.0) / (exp_2a + 1.0)

    y = 0.5 * x * (1.0 + tanh_a)
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 4.4 逐行拆解

```
x³        → x * x * x                        # 三次方
0.044715·x³  → 0.044715 * x * x * x           # 乘以常数
x + 0.044715·x³ → x + 0.044715 * x * x * x   # 相加
√(2/π)·(...) → 0.79788456 * (...)             # 乘以 √(2/π)
tanh(a) → (exp(2a) - 1) / (exp(2a) + 1)      # 展开
0.5·x·(1+tanh) → 0.5 * x * (1 + tanh_a)      # 最终计算
```

**每一步都是向量化的**，同时处理 BLOCK_SIZE 个元素。

---

## 5. 性能对比 — 为什么用 Triton？

### 5.1 三种实现方式

| 方式 | 代码量 | 运行位置 | 速度 |
|------|--------|---------|------|
| 手写 Python | ~5 行 | CPU | 🐌 最慢（基准） |
| PyTorch 原生 | ~1 行 | C++ 后端 | 🐇 快 |
| Triton kernel | ~15 行 | GPU kernel | 🚀 通常最快或接近最快 |

### 5.2 为什么手写 Python 很慢？

```python
def manual_relu(x):
    result = torch.empty_like(x)
    for i in range(x.numel()):     # ← Python 循环！
        result[i] = max(0.0, x[i])  # ← 每次迭代都有 Python 开销
    return result
```

每个循环迭代都有 Python 解释器的开销（字节码调度、类型检查等）。

### 5.3 为什么 PyTorch 原生很快？

```python
y = torch.relu(x)  # ← C++ CUDA kernel，高度优化
```

PyTorch 的 `torch.relu` 调用的是 NVIDIA CUTLASS 等高度优化的 CUDA kernel。

### 5.4 为什么 Triton 也很重要？

```python
y = triton_relu_wrapper(x)  # ← Triton kernel
```

Triton 的优势不在于单个激活函数（因为 PyTorch 已经高度优化），而在于：

1. **Kernel Fusion** — 把多个操作合成一个 kernel，减少显存读写
2. **自定义操作** — PyTorch 没有的内核，你可以自己写
3. **学习 GPU 编程** — 理解底层原理

> **核心原理：** 减少显存读写。
> 朴素写法：每个操作读一次、写一次 → N 次操作 = N 次显存读写
> Fused 写法：一个 kernel 完成所有操作 → N 次操作 = 1 次读 + 1 次写

---

## 6. 动手练习

### 练习 1：实现 Elu 激活函数

```
Elu(x) = { x,           if x > 0
         { α·(eˣ - 1),  if x ≤ 0
```

提示：用 `tl.where(condition, a, b)` 做条件选择。

### 练习 2：实现 HardSigmoid

```
HardSigmoid(x) = clip((x + 1) / 2, 0, 1)
```

提示：`tl.minimum(tl.maximum((x + 1) / 2, 0.0), 1.0)`

### 练习 3：性能实验

修改 `benchmark_comparison()` 中的张量大小，观察以下规律：

```python
for size in [1024, 10240, 102400, 1048576]:
    x = torch.randn(size, device="cuda")
    # 比较三种实现的时间
```

**思考：** 为什么 PyTorch 在小张量（1024 元素）上的优势不明显？

<details>
<summary>点击查看分析</summary>

小张量时，GPU kernel 的启动开销（kernel launch overhead）占了很大比例。
随着张量增大，计算时间增长，启动开销占比减小，PyTorch 的优化优势才显现。

Triton 的启动开销与 PyTorch 相当（两者都是 CUDA kernel），所以在大张量时两者接近。

</details>

---

## 7. 核心知识点总结

### 7.1 Triton 常用函数速查

```python
tl.maximum(a, b)     → 逐元素取 max
tl.exp(x)            → 逐元素 exp
tl.where(cond, a, b) → 条件选择: cond ? a : b
tl.sin(x)            → 逐元素 sin
tl.cos(x)            → 逐元素 cos
tl.tanh(x)           → ⚠️ 不存在！手动展开
```

### 7.2 元素级操作模板

```python
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = your_formula(x)
    tl.store(y_ptr + offsets, y, mask=mask)

# 启动
num_blocks = triton.cdiv(n, BLOCK_SIZE)
kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
```

### 7.3 tanh 手动展开公式

```
tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
```

这个公式在 Triton 中反复出现（GeLU、LayerNorm 等都用得到），建议记住。

---

## 8. 常见问题

**Q：为什么 `tl.exp(-x)` 不会溢出？**

A：实际上可能会。当 `x` 是很大的负数时，`-x` 是很大的正数，`exp(-x)` 会溢出到 `inf`。
Triton 内部做了保护（返回 `inf` 而不是 NaN），但结果可能不正确。
这就是为什么 PyTorch 的 `silu` 内部可能有额外的数值保护代码。

**Q：`tl.maximum` 和 `tl.where` 哪个更快？**

A：性能基本相同，都是向量化操作，生成相似的 PTX 代码。
`tl.maximum` 更简洁，推荐优先使用。

**Q：BLOCK_SIZE 为什么必须是 2 的幂？**

A：Triton 编译器用 BLOCK_SIZE 生成 mask 和内存访问模式。
2 的幂让编译器可以高效地生成位运算（`offset & (BLOCK_SIZE-1)` 等价于 `offset % BLOCK_SIZE`）。

---

**运行本章代码：** `python 02_elementwise.py`
**下一章：** 第三章 — 行级聚合（Softmax, LayerNorm）
