# 第一章：Triton Hello World — 你的第一个 GPU Kernel

> **你将学到什么：**
> - GPU 编程的核心抽象：Grid → Block → Thread
> - Triton 的 `@triton.jit` 装饰器
> - `tl.program_id()`, `tl.arange()`, `tl.load()`, `tl.store()`
> - 如何启动一个 kernel 并验证结果
>
> **前置知识：** 基本 Python、PyTorch 张量操作
> **预计时间：** 30 分钟
> **最终成果：** 一个能在 GPU 上加 1 的 Triton kernel

---

## 1. GPU 编程的三层抽象

### 1.1 从并行计算说起

GPU 的核心思想很简单：**让成千上万个线程同时工作，每个线程处理一个数据元素。**

想象你要给 100 万个数字各加 1：

```python
# CPU 的做法：一个接一个
for i in range(1_000_000):
    x[i] = x[i] + 1
```

GPU 的做法：**100 万个线程同时做这件事**，每个线程只处理一个元素。

### 1.2 三层组织结构

GPU 的线程不是一盘散沙，而是三层组织：

```
┌─────────────────────────────────────────────┐
│  Grid (网格) — 整个计算任务                  │
│  ┌─────────────┐  ┌─────────────┐           │
│  │ Block 0     │  │ Block 1     │  ...      │
│  │ ┌───┬───┬───┐│  │ ┌───┬───┬───┐│         │
│  │ │T0 │T1 │T2 ││  │ │T0 │T1 │T2 ││         │
│  │ │T3 │T4 │T5 ││  │ │T3 │T4 │T5 ││         │
│  │ └───┴───┴───┘│  │ └───┴───┴───┘│         │
│  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────┘
```

| 层级 | 类比 | 特点 |
|------|------|------|
| **Grid** | 整个工厂 | 包含所有 Block |
| **Block** | 一条流水线 | Block 内的线程共享高速内存 |
| **Thread** | 单个工人 | 执行一段代码，处理一个数据 |

**关键规则：**
- Block 内的线程可以互相通信（通过 Shared Memory）和同步（barrier）
- Block 之间的线程**不能**直接通信
- 每个 Thread 执行**完全相同的代码**，但处理**不同的数据索引**

### 1.3 类比：工厂流水线

> **Warehouse : DRAM :: Factory : SRAM**
>
> - DRAM（显存）= 仓库，容量大（80GB），速度慢
> - SRAM（共享内存）= 工厂工作台，容量小（几百KB），速度快
>
> Triton 帮你自动管理"从仓库拿货 → 放到工作台 → 加工 → 放回仓库"的过程。

---

## 2. 你的第一个 Triton Kernel

### 2.1 最简单的 Kernel：给每个元素加 1

```python
import triton
import triton.language as tl
import torch

@triton.jit
def hello_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 第1步：我是第几个 block?
    pid = tl.program_id(axis=0)

    # 第2步：我从哪个位置开始处理?
    block_start = pid * BLOCK_SIZE

    # 第3步：我负责哪些元素?
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 第4步：处理边界（数组可能不够长）
    mask = offsets < n_elements

    # 第5步：从显存读取
    x = tl.load(x_ptr + offsets, mask=mask)

    # 第6步：计算
    y = x + 1

    # 第7步：写回显存
    tl.store(y_ptr + offsets, y, mask=mask)
```

### 2.2 逐行解析

| 代码 | 作用 | 类比 |
|------|------|------|
| `@triton.jit` | 标记这是一个 GPU kernel | "以下代码在 GPU 上运行" |
| `pid = tl.program_id(axis=0)` | 获取当前 block 的编号 (0, 1, 2, ...) | "我是第几号流水线" |
| `block_start = pid * BLOCK_SIZE` | 计算这个 block 的起始位置 | "我从第几个元素开始" |
| `tl.arange(0, BLOCK_SIZE)` | 生成 `[0, 1, 2, ..., BLOCK_SIZE-1]` | "我负责这一批的偏移量" |
| `mask = offsets < n_elements` | 处理边界：如果数组只有 5000 个元素，第 5001-6000 个位置不处理 | "别越界！" |
| `tl.load()` | 从全局内存 (显存) 读取数据 | "从仓库拿货" |
| `tl.store()` | 写回全局内存 | "把成品放回仓库" |

### 2.3 关键概念：BLOCK_SIZE 和 constexpr

```python
BLOCK_SIZE: tl.constexpr
```

- `tl.constexpr` 告诉 Triton：**这是一个编译期常量**
- Triton 编译器会用它做优化
- BLOCK_SIZE 必须是 **2 的幂**（128, 256, 512, 1024...）
- 为什么？因为 Triton 用 power-of-2 来生成高效的 mask 和内存访问模式

**选多大的 BLOCK_SIZE？**
- 太小：kernel 启动开销占比大
- 太大：每个 block 寄存器压力太大
- **1024 是一个安全的起点**

---

## 3. 启动 Kernel：Grid 配置

### 3.1 计算需要多少个 block

```python
n_elements = 5000
BLOCK_SIZE = 1024

# 需要几个 block?
num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
# triton.cdiv = ceiling division = ceil(5000/1024) = 5
```

| n_elements | BLOCK_SIZE | num_blocks |
|------------|-----------|------------|
| 5000 | 1024 | 5 |
| 10000 | 1024 | 10 |
| 1024 | 1024 | 1 |

### 3.2 启动语法

```python
# 语法: kernel_name[(grid_config)](参数列表)
hello_kernel[(num_blocks,)](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
```

`(num_blocks,)` 是一个 tuple，表示 Grid 有 `num_blocks` 个 block。

- 1D Grid：`(num_blocks,)` — 一维排列
- 2D Grid：`(num_rows, num_cols)` — 二维排列（用于矩阵）
- 3D Grid：`(nx, ny, nz)` — 三维排列

### 3.3 包装函数

Triton kernel 不能直接被 Python 调用，需要包装：

```python
def hello_wrapper(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "输入必须在 GPU 上"
    assert x.is_contiguous(), "输入必须是连续的"

    n = x.numel()
    y = torch.empty_like(x)  # 分配输出张量

    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)

    hello_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)

    return y
```

**两个重要断言：**
- `x.is_cuda` — Triton 只支持 GPU 张量
- `x.is_contiguous()` — Triton 要求内存连续，否则 `tl.load` 无法正确工作

---

## 4. 验证：运行你的第一个 Kernel

### 4.1 基本测试

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda")
y = hello_wrapper(x)

# 输入:  [1.0, 2.0, 3.0, 4.0, 5.0]
# 输出:  [2.0, 3.0, 4.0, 5.0, 6.0]  ✅
```

### 4.2 边界测试

```python
# 非对齐长度：5000 不是 1024 的倍数
x_big = torch.randn(5000, device="cuda")
y_big = hello_wrapper(x_big)
assert torch.allclose(y_big.cpu(), x_big.cpu() + 1)  # mask 确保边界正确
```

### 4.3 看 PTX 代码（进阶）

Triton 编译后会生成 PTX（GPU 汇编语言）：

```python
import os
os.environ["TRITON_INTERPRET"] = "0"

# 运行一次以生成 PTX
y = hello_wrapper(x)

# 查看生成的 PTX
ptx = list(hello_kernel.cache.values())[0].asm["ptx"]
print(ptx)
```

PTX 中你会看到：
- `ld.global.f32` — 从全局内存加载
- `st.global.f32` — 写回全局内存
- `add.f32` — 浮点加法

---

## 5. 动手练习

### 练习 1：修改 Kernel

把 `y = x + 1` 改成以下操作，验证结果正确性：

```python
# a) y = x * 2
# b) y = x ** 2
# c) y = torch.sin(x) → 用 tl.sin(x)
```

### 练习 2：双输入 Kernel

写一个 `add_kernel(a_ptr, b_ptr, c_ptr, ...)` 实现 `c = a + b`：

```python
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)
```

### 练习 3：理解 Grid

思考以下问题：

**Q1：** 如果 `n_elements = 2048`，`BLOCK_SIZE = 1024`，会有几个 block 被启动？每个 block 处理哪些元素？

**Q2：** 如果 `n_elements = 2049`，`BLOCK_SIZE = 1024`，会发生什么？（提示：最后一个 block 的部分 thread 会被 mask 掉）

**Q3：** 为什么 `mask = offsets < n_elements` 而不是 `mask = block_start < n_elements`？

<details>
<summary>点击查看答案</summary>

**Q1：** 2 个 block。Block 0 处理 [0, 1023]，Block 1 处理 [1024, 2047]。

**Q2：** 3 个 block。Block 0: [0-1023]，Block 1: [1024-2047]，Block 2: [2048]（只有 1 个元素有效，其他 1023 个被 mask 跳过）。

**Q3：** 因为每个 thread 处理的是 `offsets` 中的单个元素，必须对每个元素单独判断是否越界。`block_start < n_elements` 只判断了 block 的起始位置，但 block 内的某些元素可能已经越界。

</details>

---

## 6. 核心公式总结

```
Triton 元素级操作模板:

  pid = tl.program_id(axis=0)
  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n_elements

  x = tl.load(ptr + offsets, mask=mask)   # 读
  y = your_computation(x)                  # 算
  tl.store(ptr + offsets, y, mask=mask)   # 写
```

---

## 7. 下一章预告

下一章，我们将用这个模板实现更多激活函数（ReLU, GeLU, Silu），并和 PyTorch 原生实现做性能对比，直观感受 Triton 的价值。

**运行本章代码：** `python 01_introduction.py`
