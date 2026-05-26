# CS336 Assignment 2: FlashAttention 详解

> **核心思想**: 通过 **Tiling (分块)**、**Recomputation (重计算)** 和 **Kernel Fusion (算子融合)**，减少 GPU 显存占用并极大提升推理速度。

---

## 🚀 1. 为什么需要 FlashAttention?

标准 Attention 算法（Scaled Dot-Product Attention）存在两大瓶颈：

### 1.1 显存占用 (Memory Footprint) - $O(N^2)$
对于长度为 $N$ 的序列，需要存储 $N \times N$ 的中间注意力分数矩阵 $S$ 和注意力概率矩阵 $P$。
-   如果 $N=1K$，矩阵大小约 4MB (FP32)。
-   如果 $N=128K$ (如 Claude 3)，矩阵大小约 **64GB**！
-   显存瞬间爆炸，根本无法训练或推理长文本。

### 1.2 内存带宽 (Memory Bandwidth) - HBM vs SRAM
GPU 内存分两级：
-   **HBM (High Bandwidth Memory)**: 容量大 (80GB)，速度慢 (2TB/s)。
-   **SRAM (Shared Memory)**: 容量小 (100KB/SM)，速度极快 (19TB/s)。

标准 Attention 需要反复读写 HBM：
1.  读 $Q, K^T$ 从 HBM $\to$ SRAM，计算 $S=QK^T$，写 $S$ 回 HBM。
2.  读 $S$ 从 HBM $\to$ SRAM，计算 $P=\text{Softmax}(S)$，写 $P$ 回 HBM。
3.  读 $P, V$ 从 HBM $\to$ SRAM，计算 $O=PV$，写 $O$ 回 HBM。

**FlashAttention 的创新**:
**直接在 SRAM 中完成所有计算**，完全不把中间矩阵 $S$ 和 $P$ 写回 HBM！

---

## 🔥 2. 核心算法：Online Softmax

要在分块计算中正确计算 Softmax，必须使用 **Online Softmax** 技巧。因为 Softmax 需要全局最大值和全局分母，而分块计算时还没看到后面的块。

### 2.1 传统 Softmax
$$ m(x) = \max_i x_i $$
$$ f(x) = \left[ e^{x_1 - m(x)}, \dots, e^{x_N - m(x)} \right] $$
$$ \ell(x) = \sum_i f(x)_i $$
$$ \text{Softmax}(x) = \frac{f(x)}{\ell(x)} $$
需要遍历所有 $x_i$ 两次（一次求 max，一次求 sum）。

### 2.2 Online Softmax 更新公式
我们可以维护两个运行中的统计量：
-   $m_{running}$: 当前遇到的局部最大值。
-   $\ell_{running}$: 当前的局部归一化因子（分母）。

当处理新的一块数据 $x_{block}$ (长度 $B$) 时：
1.  计算当前块的局部最大值 $m_{block} = \max(x_{block})$。
2.  计算当前块的局部指数和 $\ell_{block} = \sum e^{x_{block} - m_{block}}$。
3.  更新 **全局最大值**:
    $$ m_{new} = \max(m_{running}, m_{block}) $$
4.  更新 **全局归一化因子**:
    $$ \ell_{new} = \ell_{running} \cdot e^{m_{running} - m_{new}} + \ell_{block} \cdot e^{m_{block} - m_{new}} $$
    *(注意：原本的 $\ell_{running}$ 是基于旧的 $m_{running}$ 归一化的，现在要统一缩放到 $m_{new}$ 下)*

---

## 🧩 3. FlashAttention Tiling 算法流程

我们把 $Q, K, V$ 切分成块：$Q$ 分为 $T_r$ 块，$K, V$ 分为 $T_c$ 块。每块大小为 $B_r \times d$ 和 $B_c \times d$。

### 外层循环 (Loop over Query Blocks $Q_i$)
加载 $Q_i$ 从 HBM 到 SRAM。
初始化输出 $O_i = 0$，统计量 $\ell_i = 0, m_i = -\infty$。

### 内层循环 (Loop over Key/Value Blocks $K_j, V_j$)
1.  加载 $K_j, V_j$ 从 HBM 到 SRAM。
2.  **计算分数**: $S_{ij} = Q_i K_j^T$ (在 SRAM 上计算)。
3.  **更新统计量**:
    -   $m_{ij} = \text{rowmax}(S_{ij})$
    -   $P_{ij} = \exp(S_{ij} - m_{ij})$  *(未归一化的概率)*
    -   $\ell_{ij} = \text{rowsum}(P_{ij})$
4.  **更新全局最大值**:
    -   $m_{new} = \max(m_i, m_{ij})$
    -   $\ell_{new} = \ell_i \cdot e^{m_i - m_{new}} + \ell_{ij} \cdot e^{m_{ij} - m_{new}}$
5.  **更新输出 $O_i$**:
    -   把旧的 $O_i$ 重新缩放 (rescale) 到新的最大值下：
        $$ O_i \leftarrow O_i \cdot \frac{\ell_i \cdot e^{m_i - m_{new}}}{\ell_{new}} + \frac{P_{ij} V_j \cdot e^{m_{ij} - m_{new}}}{\ell_{new}} $$
6.  更新指针: $\ell_i \leftarrow \ell_{new}, m_i \leftarrow m_{new}$。

### 结束循环
写回 $O_i$ 到 HBM。

---

## 💻 4. 代码实现 (Triton 伪代码)

这就是 `get_flashattention_autograd_function_triton` 需要实现的内核逻辑。

```python
import triton
import triton.language as tl

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,  # 指针
    stride_qm, stride_kn, ...  # 步长
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ...
):
    # 1. 指针初始化 (Block Pointers)
    # ... (省略指针算术)

    # 2. 加载 Q 到 SRAM
    q = tl.load(q_ptrs)

    # 3. 初始化累加器
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DHEAD], dtype=tl.float32)

    # 4. 循环遍历 K, V 块
    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # 计算 QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        # Online Softmax 更新
        m_ij = tl.max(qk, 1)        # 局部最大值
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)         # 局部 sum
        
        # 更新全局统计量
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)
        
        l_new = alpha * l_i + beta * l_ij
        
        # 更新输出 acc (重新缩放旧的部分 + 加上新的部分)
        # acc = acc * alpha * (l_i / l_new) + (p @ v) * beta * (1 / l_new)
        # 为了数值稳定通常合并缩放因子
        p_scale = beta / l_new
        acc_scale = l_i / l_new * alpha
        
        acc = acc * acc_scale[:, None] + tl.dot(p.to(tl.float16), v) * p_scale[:, None]
        
        # 更新状态
        l_i = l_new
        m_i = m_new

        # 移动 K, V 指针
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # 5. 写回 HBM
    tl.store(out_ptrs, acc)
```

---

## 🔗 5. 与 vLLM PagedAttention 的联系

**PagedAttention** 是 FlashAttention 在内存管理上的延伸：

-   **FlashAttention**: 假设 $K, V$ 存在连续的内存块中。
-   **PagedAttention**: 允许 $K, V$ 被切分成固定大小的 Page (如 16 tokens)，存储在非连续的物理内存中。

**计算逻辑是完全一样的**！
-   FlashAttention 的内层循环是 `k_ptrs += stride` (线性移动)。
-   PagedAttention 的内层循环是 `block_idx = page_table[batch_idx, logical_block_idx]` (查表获取物理地址)。

理解并实现了 Assignment 2 的 FlashAttention，您就完全具备了理解 vLLM 内核 (`vllm/attention/ops/triton/`) 的能力。
