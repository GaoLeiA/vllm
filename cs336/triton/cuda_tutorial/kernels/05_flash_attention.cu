/*
 * CUDA 从零开始教程 — 第五章: FlashAttention Forward
 * ====================================================
 *
 * 对应 Triton 教程: triton_tutorial/05_flash_attention.py
 *
 * 这是最复杂的一个 kernel，综合了:
 *   1. Shared Memory (第三章)
 *   2. Parallel Reduction (第三章)
 *   3. Tiling (第四章)
 *   4. Online Softmax (新概念)
 *
 * 设计:
 *   Grid:  (Nq, B) — 每个 Block 处理一个 query 行
 *   Block: D 个线程 — 每个线程处理一个维度
 *
 *   每个 Block 的工作:
 *     1. 加载 Q[i,:] 到寄存器 (每个线程存一个 q_val)
 *     2. 循环遍历 KV blocks:
 *        a. 计算 S[i,j] = Q[i,:] @ K[j,:]^T  (parallel reduction!)
 *        b. Online Softmax 更新
 *        c. O[i,:] += P[j] * V[j,:]
 *     3. 最终归一化: O /= L
 *
 * 对比 Triton:
 *   Triton: tl.dot(Q_i, K_j.T) ← 一行代码
 *   CUDA:   手动 dot product + shared memory reduction
 *
 *   Triton: tl.max(S_ij, axis=1) ← 一行代码
 *   CUDA:   再一次 shared memory reduction
 *
 *   这就是为什么 FlashAttention 的 CUDA 实现比 Triton 长很多!
 */

#include <torch/extension.h>
#include "common.h"

#define FA_K_BLOCK 32  // 每次处理 32 个 KV 位置

// ============================================================
// FlashAttention Forward Kernel
// ============================================================
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,    // (B, Nq, D)
    const float* __restrict__ K,    // (B, Nk, D)
    const float* __restrict__ V,    // (B, Nk, D)
    float* __restrict__ O,          // (B, Nq, D)
    int Nq, int Nk, int D,
    float scale,
    bool is_causal
) {
    int q_idx = blockIdx.x;     // 当前处理第几个 query
    int batch_idx = blockIdx.y; // 当前处理第几个 batch
    int d = threadIdx.x;        // 当前线程处理第几个维度

    if (q_idx >= Nq) return;

    // ---- Step 1: 指针定位 ----
    const float* q_row = Q + (batch_idx * Nq + q_idx) * D;
    const float* k_base = K + batch_idx * Nk * D;
    const float* v_base = V + batch_idx * Nk * D;
    float* o_row = O + (batch_idx * Nq + q_idx) * D;

    // 每个线程存储自己维度的 Q 值
    float q_val = (d < D) ? q_row[d] : 0.0f;

    // ---- Step 2: Shared Memory 分配 ----
    // s_reduce: 用于 dot product 的 parallel reduction
    // s_scores: 存储当前 KV block 的 attention scores
    extern __shared__ float smem[];
    float* s_reduce = smem;                     // [blockDim.x]
    float* s_scores = smem + blockDim.x;        // [FA_K_BLOCK]

    // ---- Step 3: 初始化 Online Softmax 状态 ----
    // 对应 Triton 的:
    //   M_acc = tl.full((Q_BLOCK, 1), float("-inf"), ...)
    //   L_acc = tl.zeros(...)
    //   O_acc = tl.zeros(...)
    float m_acc = -INFINITY;  // running max
    float l_acc = 0.0f;       // running sum (归一化因子)
    float o_acc = 0.0f;       // running output (当前线程的维度 d)

    // Causal: 只看 j <= q_idx 的位置
    int k_limit = is_causal ? min(Nk, q_idx + 1) : Nk;

    // ---- Step 4: 遍历 KV blocks ----
    // 对应 Triton 的: for k_block_id in range(tl.cdiv(Nk, K_BLOCK)):
    for (int k_start = 0; k_start < k_limit; k_start += FA_K_BLOCK) {
        int k_end = min(k_start + FA_K_BLOCK, k_limit);
        int k_len = k_end - k_start;

        // ---- Step 4.1: 计算 attention scores ----
        // S[j] = Q[i,:] @ K[j,:]^T * scale
        // 在 Triton 中: S_ij = tl.dot(Q_i, K_j.T) * scale
        // 在 CUDA 中: 需要手动做 dot product + reduction
        for (int j = 0; j < k_len; j++) {
            const float* k_row = k_base + (k_start + j) * D;

            // 每个线程计算一个 partial product
            s_reduce[d] = (d < D) ? (q_val * k_row[d] * scale) : 0.0f;
            __syncthreads();

            // 树形归约求和 (parallel reduction)
            // 对应 Triton 自动做的 dot product
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (d < s) {
                    s_reduce[d] += s_reduce[d + s];
                }
                __syncthreads();
            }

            // 线程 0 存储最终的 score
            if (d == 0) s_scores[j] = s_reduce[0];
            __syncthreads();
        }

        // ---- Step 4.2: 找当前 block 的 max ----
        // 对应 Triton 的: M_block = tl.max(S_ij, axis=1)
        // 所有线程读 shared memory 中的 s_scores，独立计算 max
        float m_block = -INFINITY;
        for (int j = 0; j < k_len; j++) {
            m_block = fmaxf(m_block, s_scores[j]);
        }

        // ---- Step 4.3: Online Softmax 更新 ----
        // 对应 Triton 的:
        //   M_new = tl.maximum(M_acc, M_block)
        //   alpha = exp(M_acc - M_new)
        //   L_new = alpha * L + sum(P) * exp(M_block - M_new)
        //   O_new = alpha * O + P @ V
        float m_new = fmaxf(m_acc, m_block);
        float alpha = expf(m_acc - m_new);

        // 修正旧的累加值
        o_acc *= alpha;
        l_acc *= alpha;

        // 加入新 block 的贡献
        for (int j = 0; j < k_len; j++) {
            float p_j = expf(s_scores[j] - m_new);
            const float* v_row = v_base + (k_start + j) * D;

            // O += P @ V (每个线程只更新自己的维度 d)
            if (d < D) {
                o_acc += p_j * v_row[d];
            }
            // L += sum(P)
            l_acc += p_j;
        }

        m_acc = m_new;
    }

    // ---- Step 5: 最终归一化 ----
    // 对应 Triton 的: O_i = O_acc / L_acc
    if (d < D) {
        o_row[d] = o_acc / l_acc;
    }
}

// ============================================================
// 包装函数
// ============================================================
torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K,
                                  torch::Tensor V, bool is_causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
    TORCH_CHECK(Q.dim() == 3, "输入形状必须是 (B, N, D)");

    int B = Q.size(0);
    int Nq = Q.size(1);
    int D = Q.size(2);
    int Nk = K.size(1);

    auto O = torch::empty_like(Q);
    float scale = 1.0f / sqrtf((float)D);

    // blockDim.x = next_power_of_2(D)，保证 reduction 正确
    int block_dim = next_pow2(D);
    TORCH_CHECK(block_dim <= 1024, "D 太大，超出线程数限制");

    dim3 grid(Nq, B);
    dim3 block(block_dim);

    // Shared memory: s_reduce[block_dim] + s_scores[FA_K_BLOCK]
    int smem_size = (block_dim + FA_K_BLOCK) * sizeof(float);

    flash_attn_fwd_kernel<<<grid, block, smem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), Nq, Nk, D, scale, is_causal);

    return O;
}
