/*
 * CUDA 从零开始教程 — 第一章: 环境与基本概念
 * =============================================
 *
 * 对应 Triton 教程: triton_tutorial/01_introduction.py
 *
 * Triton vs CUDA 的关键区别:
 *   Triton: 你管理 Block，编译器管理 Thread
 *   CUDA:   你管理 Grid + Block + Thread 所有层级
 *
 * Triton 的 tl.program_id(0) → CUDA 的 blockIdx.x
 * Triton 的 tl.arange(0, BLOCK_SIZE) → CUDA 的 threadIdx.x
 * Triton 的 mask → CUDA 的 if (idx < n)
 */

#include <torch/extension.h>
#include "common.h"

// ============================================================
// CUDA Kernel: 向量加 1
// ============================================================
/*
 * 对比 Triton 版本:
 *
 *   Triton:
 *     pid = tl.program_id(0)
 *     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
 *     mask = offsets < n
 *     x = tl.load(x_ptr + offsets, mask=mask)
 *     tl.store(y_ptr + offsets, x + 1, mask=mask)
 *
 *   CUDA:
 *     idx = blockIdx.x * blockDim.x + threadIdx.x
 *     if (idx < n) y[idx] = x[idx] + 1
 *
 * 关键区别:
 *   - Triton 中一个 Block 的所有线程共享同一段代码 (SIMT)，
 *     Triton 编译器自动把 tl.arange 展开到每个线程
 *   - CUDA 中你需要手动算每个线程的全局索引
 *   - Triton 用 mask 处理边界，CUDA 用 if 判断
 */
__global__ void hello_kernel(
    const float* __restrict__ x,   // 输入指针 (全局内存)
    float* __restrict__ y,         // 输出指针 (全局内存)
    int n                          // 元素总数
) {
    // 计算当前线程的全局索引
    // blockIdx.x  = 第几个 Block   (对应 Triton 的 tl.program_id(0))
    // blockDim.x  = 每个 Block 的线程数 (对应 Triton 的 BLOCK_SIZE)
    // threadIdx.x = Block 内的第几个线程 (对应 Triton 的 tl.arange 中的某个值)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查 (对应 Triton 的 mask = offsets < n_elements)
    if (idx < n) {
        y[idx] = x[idx] + 1.0f;
    }
}

// ============================================================
// 包装函数: 把 CUDA Kernel 包装成 PyTorch 可调用的函数
// ============================================================
/*
 * 对比 Triton 版本:
 *
 *   Triton:
 *     num_blocks = triton.cdiv(n, block_size)
 *     hello_kernel[(num_blocks,)](x, y, n, BLOCK_SIZE=block_size)
 *
 *   CUDA:
 *     hello_kernel<<<num_blocks, block_size>>>(x_ptr, y_ptr, n)
 *
 * 语法区别:
 *   Triton: kernel[(grid,)](args, CONSTEXPR=val)
 *   CUDA:   kernel<<<grid, block>>>(args)
 *
 * Triton 的 grid 只指定 Block 数量
 * CUDA 的 <<<grid, block>>> 同时指定 Grid 和 Block 的维度
 */
torch::Tensor hello_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "输入必须在 GPU 上");
    TORCH_CHECK(x.is_contiguous(), "输入必须是连续的");

    int n = x.numel();
    auto y = torch::empty_like(x);

    // Grid 配置 (与 Triton 一一对应)
    int block_size = 1024;                    // 每个 Block 的线程数
    int num_blocks = ceil_div(n, block_size);  // Block 数量

    // 启动 Kernel
    // <<<num_blocks, block_size>>> 是 CUDA 特有的语法
    hello_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}
