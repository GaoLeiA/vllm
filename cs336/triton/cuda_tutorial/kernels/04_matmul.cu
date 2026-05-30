/*
 * CUDA 从零开始教程 — 第四章: 矩阵乘法 (Tiled Matrix Multiplication)
 * ==================================================================
 *
 * 对应 Triton 教程: triton_tutorial/04_matmul.py
 *
 * 核心概念: Shared Memory Tiling
 *
 * 朴素 MatMul 的问题:
 *   C[i][j] = sum_k A[i][k] * B[k][j]
 *   每个 C 元素需要读 K 次 A 和 K 次 B → 2*M*N*K 次全局内存读取
 *
 * Tiled MatMul:
 *   把 A, B 切成 TILE×TILE 的小块
 *   每次从全局内存加载一个小块到 Shared Memory (SRAM)
 *   在 SRAM 中完成计算 → 大幅减少全局内存读取
 *
 * Triton vs CUDA:
 *   Triton: tl.dot(a, b) + tl.load() 自动利用 shared memory
 *   CUDA:   手动声明 __shared__，手动 __syncthreads()
 *
 * 内存层次 (速度从快到慢):
 *   Register (寄存器)  → ~0 cycles
 *   Shared Memory (SRAM) → ~5 cycles    ← 我们手动管理这一层
 *   L2 Cache             → ~100 cycles
 *   Global Memory (HBM)  → ~400 cycles
 */

#include <torch/extension.h>
#include "common.h"

#define TILE_SIZE 32

// ============================================================
// Tiled MatMul Kernel
// ============================================================
/*
 * Grid:  (ceil(N/TILE), ceil(M/TILE)) — 二维 Grid!
 * Block: (TILE, TILE) = (32, 32) = 1024 个线程
 * 每个线程计算 C 的一个元素
 *
 * 对比 Triton:
 *   Triton 用 2D program_id:
 *     pid_m = tl.program_id(0)
 *     pid_n = tl.program_id(1)
 *
 *   CUDA 用 2D blockIdx:
 *     blockIdx.y → M 维度
 *     blockIdx.x → N 维度
 *
 *   Triton 的 tl.dot(a, b) 内部也使用 shared memory,
 *   但编译器帮你管理了! 在 CUDA 中你需要:
 *     1. 声明 __shared__ 数组
 *     2. 手动加载数据
 *     3. 手动 __syncthreads()
 *     4. 手动计算
 */
__global__ void matmul_tiled_kernel(
    const float* __restrict__ A,   // (M, K)
    const float* __restrict__ B,   // (K, N)
    float* __restrict__ C,         // (M, N)
    int M, int N, int K
) {
    // ---- Step 1: 声明 Shared Memory ----
    // 每个 Block 有自己的 shared memory (Block 内所有线程共享)
    // 注意: 这是 Triton 不需要你写的部分!
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // ---- Step 2: 确定当前线程处理 C 的哪个元素 ----
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // C 的行
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // C 的列

    float sum = 0.0f;

    // ---- Step 3: 遍历 K 维度的 tile ----
    // 每次加载 A 和 B 的一个 TILE×TILE 子块
    int num_tiles = ceil_div(K, TILE_SIZE);

    for (int t = 0; t < num_tiles; t++) {
        // ---- Step 3.1: 从全局内存加载到 Shared Memory ----
        // 每个线程加载一个元素 (合作加载!)
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // 边界检查
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // ---- Step 3.2: 同步! ----
        // 等所有线程加载完毕后才能开始计算
        // 这是 CUDA 手动管理 shared memory 的代价!
        __syncthreads();

        // ---- Step 3.3: 在 Shared Memory 中计算 ----
        // 这一步对应 Triton 的 acc += tl.dot(a, b)
        // 但我们是手动展开的
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ---- Step 3.4: 再次同步! ----
        // 确保所有线程计算完毕后再加载下一个 tile
        __syncthreads();
    }

    // ---- Step 4: 写回全局内存 ----
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================
// 包装函数
// ============================================================
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.is_contiguous());
    TORCH_CHECK(B.is_cuda() && B.is_contiguous());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2);
    TORCH_CHECK(A.size(1) == B.size(0), "K 维度不匹配");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // 2D Grid 配置 (对比 Triton 的 grid = (num_m_blocks, num_n_blocks))
    dim3 block(TILE_SIZE, TILE_SIZE);  // 32×32 = 1024 线程
    dim3 grid(ceil_div(N, TILE_SIZE), ceil_div(M, TILE_SIZE));

    matmul_tiled_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, N, K);

    return C;
}
