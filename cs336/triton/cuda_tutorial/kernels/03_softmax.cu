/*
 * CUDA 从零开始教程 — 第三章: 行级聚合操作 (Row-wise Aggregation)
 * ================================================================
 *
 * 对应 Triton 教程: triton_tutorial/03_softmax.py
 *
 * 核心概念: Shared Memory + Parallel Reduction
 *
 * Triton vs CUDA:
 *   Triton: tl.max(x, axis=0) / tl.sum(x, axis=0)
 *           → 编译器自动生成 shared memory reduction 代码
 *
 *   CUDA:   你需要手动写 shared memory reduction!
 *           1. 每个线程计算自己的局部值
 *           2. 存入 shared memory
 *           3. 用树形归约 (tree reduction) 合并
 *           4. __syncthreads() 同步
 *
 * 这就是 Triton 相比 CUDA 最大的优势之一:
 *   reduction 操作在 Triton 中只需一行，
 *   在 CUDA 中需要 ~10 行！
 */

#include <torch/extension.h>
#include "common.h"

// ============================================================
// Softmax Kernel (使用 Shared Memory Reduction)
// ============================================================
/*
 * 设计:
 *   - 每个 Block 处理一行 (与 Triton 相同: pid → row_idx)
 *   - blockDim.x = 256 个线程协作处理一行
 *   - 每个线程处理 ceil(N/256) 个元素
 *   - 使用 shared memory 做 max 和 sum 的 reduction
 *
 * 对比 Triton:
 *   Triton:  x_max = tl.max(x_row, axis=0)    ← 一行搞定!
 *   CUDA:    需要手动写树形归约 (下面的 for 循环)
 */
__global__ void softmax_kernel(
    const float* __restrict__ x,    // 输入: (M, N)
    float* __restrict__ y,          // 输出: (M, N)
    int M,                          // 行数
    int N                           // 列数
) {
    // 当前处理第几行 (对应 Triton 的 row_idx = tl.program_id(0))
    int row = blockIdx.x;
    if (row >= M) return;

    // Shared memory 用于 reduction (动态分配)
    extern __shared__ float sdata[];

    const float* x_row = x + row * N;
    float* y_row = y + row * N;

    // ---- 第1遍: 找最大值 (数值稳定) ----
    // 每个线程先找自己负责的元素中的最大值
    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, x_row[col]);
    }

    // 存入 shared memory
    sdata[threadIdx.x] = local_max;
    __syncthreads();  // 等待所有线程写完 (Triton 不需要这一步!)

    // 树形归约: 每次减半，O(log n) 步
    // 对比 Triton: tl.max(x_row, axis=0) ← 编译器自动做这一步
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];  // 归约结果在 sdata[0]
    __syncthreads();

    // ---- 第2遍: 计算 exp(x - max) 并求和 ----
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        local_sum += expf(x_row[col] - row_max);
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // 同样的树形归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // ---- 第3遍: 归一化 ----
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        y_row[col] = expf(x_row[col] - row_max) / row_sum;
    }
}

// ============================================================
// LayerNorm Kernel
// ============================================================
/*
 * LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * 和 Softmax 使用完全相同的 reduction 模式:
 *   1. 求 mean (sum → reduction → 除以 N)
 *   2. 求 variance (diff² sum → reduction → 除以 N)
 *   3. 归一化
 */
__global__ void layernorm_kernel(
    const float* __restrict__ x,      // 输入: (M, N)
    float* __restrict__ y,            // 输出: (M, N)
    const float* __restrict__ gamma,  // 缩放参数: (N,)
    const float* __restrict__ beta,   // 偏移参数: (N,)
    int M, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float sdata[];

    const float* x_row = x + row * N;
    float* y_row = y + row * N;

    // ---- 第1遍: 计算 mean ----
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        local_sum += x_row[col];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean = sdata[0] / (float)N;
    __syncthreads();

    // ---- 第2遍: 计算 variance ----
    float local_var = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float diff = x_row[col] - mean;
        local_var += diff * diff;
    }
    sdata[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float var = sdata[0] / (float)N;
    float inv_std = rsqrtf(var + eps);  // rsqrtf = 1/sqrt (CUDA 内建函数)
    __syncthreads();

    // ---- 第3遍: 归一化 + affine 变换 ----
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float norm = (x_row[col] - mean) * inv_std;
        y_row[col] = norm * gamma[col] + beta[col];
    }
}

// ============================================================
// 包装函数
// ============================================================
torch::Tensor softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "输入必须是 2D 矩阵");

    int M = x.size(0);
    int N = x.size(1);
    auto y = torch::empty_like(x);

    int block_size = 256;  // 每行用 256 个线程协作
    int smem_size = block_size * sizeof(float);

    // 每行一个 Block (与 Triton 相同: grid = (M,))
    softmax_kernel<<<M, block_size, smem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), M, N);

    return y;
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor gamma,
                                 torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2, "输入必须是 2D 矩阵");

    int M = x.size(0);
    int N = x.size(1);
    auto y = torch::empty_like(x);

    int block_size = 256;
    int smem_size = block_size * sizeof(float);

    layernorm_kernel<<<M, block_size, smem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        M, N, (float)eps);

    return y;
}
