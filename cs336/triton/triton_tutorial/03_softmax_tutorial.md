# 第三章：行级聚合操作 — Softmax 和 LayerNorm

> **你将学到什么：**
> - 理解"一个 Block 处理一行"的编程模式
> - 实现 Triton Softmax kernel（两遍扫描）
> - 理解数值稳定性的问题（为什么减 max）
> - 实现 LayerNorm kernel
> - 理解 `tl.max()`, `tl.sum()` 的 axis 参数
>
> **前置知识：** 第二章（元素级操作模板）
> **预计时间：** 45 分钟
> **最终成果：** Triton Softmax + LayerNorm kernel

---

## 1. 元素级操作 vs 聚合操作

### 1.1 回顾：元素级操作

```
输入: [1, 2, 3, 4, 5]    每个元素独立
输出: [2, 3, 4, 5, 6]    ReLU 或 x+1，每个结果只依赖自己的输入
```

### 1.2 新的类型：聚合操作

```
输入: [1, 2, 3, 4, 5]    每个结果依赖整行
输出: [0.01, 0.03, 0.08, 0.23, 0.66]   Softmax，需要看到所有元素才能计算
```

**聚合操作的难点：** 你必须先遍历整行，找到一些统计量（max、sum），然后才能计算最终结果。这意味着**至少两遍扫描**。

### 1.3 三种聚合操作的对比

| 操作 | 需要什么统计量 | 几遍扫描 |
|------|-------------|---------|
| **Softmax** | max, sum | 2 遍（找 max → 归一化） |
| **LayerNorm** | mean, var | 2 遍（找 mean/var → normalize） |
| **LayerSum** | sum | 1 遍 |

---

## 2. Softmax — 两遍扫描的典型

### 2.1 标准 Softmax 公式

```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

问题在于**分母**：你必须先知道所有 `exp(xⱼ)` 的和，才能算任何一个 `softmax(xᵢ)`。

### 2.2 数值稳定性问题

```python
x = [100, 101, 102]
exp(x) = [2.7e43, 7.4e43, 2.0e44]  # 接近浮点数上限
exp(x).sum() = 3.0e44               # 可能溢出!
```

**解决方案：减去每行的最大值**

```
softmax(xᵢ) = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
```

减同一个值不影响结果，但让最大的 exp 变成 exp(0)=1，避免溢出。

```
x = [100, 101, 102], max = 102
x - max = [-2, -1, 0]
exp(x - max) = [0.135, 0.368, 1.0]  ← 全在安全范围内
```

### 2.3 Triton 实现（两遍扫描）

```python
@triton.jit
def softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)      # ← 当前处理第几行
    col_offsets = tl.arange(0, BLOCK_SIZE)  # ← 所有列的偏移

    # 1. 定位到这一行的起始指针
    row_start = x_ptr + row_idx * x_row_stride

    # 2. 加载整行 (带 mask)
    x_row = tl.load(row_start + col_offsets,
                    mask=col_offsets < num_cols,
                    other=float("-inf"))

    # 第1遍扫描：找 max（数值稳定）
    x_max = tl.max(x_row, axis=0)
    x_row = x_row - x_max                # 减去 max

    # 第2遍扫描：exp, sum, normalize
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # 3. 写回
    y_row_start = y_ptr + row_idx * y_row_stride
    tl.store(y_row_start + col_offsets, y_row,
             mask=col_offsets < num_cols)
```

### 2.4 关键 API：`tl.max(x, axis=0)` 和 `tl.sum(x, axis=0)`

```
tl.max(x_row, axis=0)  → 把一行的所有元素聚合成一个标量 (max)
tl.sum(x_row, axis=0)  → 把一行的所有元素聚合成一个标量 (sum)
```

`axis=0` 表示**沿着这个向量的方向聚合**，结果是标量。

### 2.5 Grid 配置

```python
# (M, N) 的矩阵
softmax_kernel[(M,)](...)
```

- **1 个 block 处理 1 行**
- 共 M 行 → M 个 block
- `program_id(0)` = 行号

### 2.6 BLOCK_SIZE 的选择

```python
block_size = triton.next_power_of_2(N)
```

因为一行有 N 个元素，BLOCK_SIZE 必须是 N 的上界且是 2 的幂。

**为什么 `next_power_of_2` 而不是 `cdiv`？**
- `cdiv(7, 4) = 2` → 这用于计算 block 数量
- `next_power_of_2(7) = 8` → 这用于 block 大小（必须是 2 的幂）

---

## 3. 手动验证 Softmax 正确性

### 3.1 用 NumPy 验证

```python
import numpy as np

def numpy_softmax(x):
    x = x - np.max(x)          # 数值稳定
    e = np.exp(x)
    return e / e.sum()

# 测试
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(numpy_softmax(x))
# [0.01165623 0.03168492 0.08612854 0.23412166 0.63640866]
# 所有元素之和 = 1.0 ✅
```

### 3.2 验证 Triton 实现

```python
x = torch.randn(100, 256, device="cuda")
y_torch = torch.softmax(x, dim=-1)   # PyTorch 参考
y_triton = softmax_wrapper(x)         # 你的实现

assert torch.allclose(y_torch, y_triton, atol=1e-5)
```

### 3.3 极端值测试

```python
x_extreme = torch.tensor([[1.0, -100.0, 100.0, -50.0]], device="cuda")
```

测试极端值确保你的实现没有溢出或产生 NaN。

---

## 4. LayerNorm — Softmax 的亲戚

### 4.1 LayerNorm 公式

```
LayerNorm(x) = (x - mean(x)) / std(x) * γ + β

其中:
  mean(x) = Σᵢ xᵢ / N
  var(x)  = Σᵢ (xᵢ - mean)² / N
  std(x)  = √(var(x) + ε)
```

### 4.2 和 Softmax 的对比

| | Softmax | LayerNorm |
|---|---------|-----------|
| **聚合统计量** | max, sum(exp) | mean, var |
| **扫描次数** | 2 | 2 |
| **计算类型** | 非线性（exp, div） | 线性（sub, mul, div） |
| **额外参数** | 无 | γ, β（learnable）|

### 4.3 Triton 实现

```python
@triton.jit
def layernorm_kernel(x_ptr, y_ptr, gamma_ptr, beta_ptr,
                     row_stride, num_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row_start = x_ptr + row_idx * row_stride
    x_row = tl.load(row_start + col_offsets,
                    mask=col_offsets < num_cols, other=0.0)

    # 计算 mean
    mean = tl.sum(x_row, axis=0) / num_cols

    # 计算 var
    var = tl.sum((x_row - mean) * (x_row - mean), axis=0) / num_cols

    # Normalize
    x_norm = (x_row - mean) / tl.sqrt(var + eps)

    # 加载 gamma 和 beta
    gamma = tl.load(gamma_ptr + col_offsets, mask=col_offsets < num_cols)
    beta = tl.load(beta_ptr + col_offsets, mask=col_offsets < num_cols)

    # 缩放平移
    y_row = x_norm * gamma + beta

    # 写回
    y_row_start = y_ptr + row_idx * row_stride
    tl.store(y_row_start + col_offsets, y_row, mask=col_offsets < num_cols)
```

### 4.4 理解 `other=0.0` 参数

```python
x_row = tl.load(row_start + col_offsets,
                mask=col_offsets < num_cols, other=0.0)
```

`other` 指定**被 mask 掉的元素用什么值填充**。

- Softmax 用 `other=float("-inf")` — 减 max 后变成 `-inf`，exp 后是 0
- LayerNorm 用 `other=0.0` — 加 0 不影响 mean/var 计算

---

## 5. 动手练习

### 练习 1：GroupSoftmax

实现按组做 softmax（不是整行，而是每 K 个元素一组）：

```python
# 输入: [1, 2, 3, 4, 5, 6]
# group_size = 2
# 输出: softmax([1,2]), softmax([3,4]), softmax([5,6])
#     = [0.27, 0.73], [0.27, 0.73], [0.27, 0.73]
```

提示：在 kernel 中用 `col_offsets % group_size` 判断每组的边界。

### 练习 2：RMSNorm

RMSNorm 是 LayerNorm 的简化版（去掉 mean）：

```
RMSNorm(x) = x / √(Σx²/N) * γ
```

### 练习 3：对比实验

```python
for N in [64, 128, 256, 512, 1024, 2048]:
    x = torch.randn(1000, N, device="cuda")
    # 对比 PyTorch softmax 和 Triton softmax 的时间
```

**思考：** 什么时候 Triton 比 PyTorch 快？什么时候差不多？

<details>
<summary>点击查看分析</summary>

- 小 N (64-256): PyTorch 更快。PyTorch 的 softmax 经过高度优化，Triton 的启动开销相对更大。
- 中 N (512-2048): 两者接近。
- 大 N (4096+): Triton 可能更快（如果 kernel 写得好的话）。

实际中，Softmax 通常由 PyTorch 原生实现，因为已经非常优化了。
Triton 的真正价值在更复杂的操作（如 FlashAttention）上。

</details>

---

## 6. 核心知识点总结

### 6.1 行级聚合模板

```python
@triton.jit
def kernel(x_ptr, y_ptr, row_stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # 1. 加载整行
    x = tl.load(x_ptr + row * row_stride + offsets,
                mask=offsets < N, other=...)

    # 2. 聚合（多遍扫描）
    stat1 = tl.max(x, axis=0)        # 第1遍
    x = x - stat1                    # 预处理
    stat2 = tl.sum(x, axis=0)        # 第2遍
    y = x / stat2                    # 归一化

    # 3. 写回
    tl.store(y_ptr + row * row_stride + offsets, y, mask=offsets < N)

# 启动: kernel[(M,)](...)  # M = 行数
```

### 6.2 `tl.max` 和 `tl.sum` 的 axis 参数

```python
# 对于 1D 向量 x (shape: BLOCK_SIZE)
tl.max(x, axis=0)    → 标量 (max of all elements)
tl.sum(x, axis=0)    → 标量 (sum of all elements)

# 对于 2D 矩阵 x (shape: M, N)
tl.max(x, axis=0)    → (N,) 每列的 max
tl.max(x, axis=1)    → (M,) 每行的 max
```

### 6.3 数值稳定三板斧

```python
# 1. 减 max
x = x - tl.max(x, axis=0)

# 2. exp
e = tl.exp(x)

# 3. 除以 sum
y = e / tl.sum(e, axis=0)
```

这个模式在 Softmax、LayerNorm、Attention 中反复出现。

---

## 7. 下一章预告

下一章是本章的进阶：矩阵乘法。Softmax 中我们学的是"一行内的聚合"，矩阵乘法需要"跨行的聚合"——这正是 Triton 最核心的编程模式：**Tiling + Shared Memory**。

**运行本章代码：** `python 03_softmax.py`
**下一章：** 第四章 — 矩阵乘法 Tiling
