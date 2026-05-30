/*
 * CUDA 从零开始教程 — 第二章: 元素级操作 (Elementwise Operations)
 * ================================================================
 *
 * 对应 Triton 教程: triton_tutorial/02_elementwise.py
 *
 * 元素级操作是最简单的 CUDA 编程模式:
 *   每个线程处理一个元素，线程之间没有通信
 *
 * 对比 Triton:
 *   模式完全相同，区别在于:
 *   - Triton 自动生成每个线程的偏移量 (tl.arange)
 *   - CUDA 需要手动计算 (blockIdx.x * blockDim.x + threadIdx.x)
 *   - Triton 用 mask，CUDA 用 if
 *   - 计算逻辑完全一样 (fmaxf, expf, tanhf 等)
 */

#include <torch/extension.h>
#include "common.h"

// ============================================================
// ReLU: y = max(0, x)
// ============================================================
/*
 * 对比 Triton:
 *   y = tl.maximum(x, 0.0)
 *
 * CUDA:
 *   y[idx] = fmaxf(x[idx], 0.0f)
 *
 * fmaxf 是 CUDA 的数学函数，对应 C 的 fmax
 */
__global__ void relu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

// ============================================================
// Silu (Swish): y = x * sigmoid(x)
// ============================================================
/*
 * sigmoid(x) = 1 / (1 + exp(-x))
 *
 * 对比 Triton:
 *   sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
 *   y = x * sigmoid_x
 *
 * CUDA 完全一样，用 expf() 替代 tl.exp()
 */
__global__ void silu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sigmoid_val = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid_val;
    }
}

// ============================================================
// GeLU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================
/*
 * 对比 Triton:
 *   a = 0.79788456 * (x + 0.044715 * x * x * x)
 *   exp_2a = tl.exp(2 * a)
 *   tanh_a = (exp_2a - 1.0) / (exp_2a + 1.0)
 *   y = 0.5 * x * (1.0 + tanh_a)
 *
 * CUDA 优势: 有原生 tanhf()，不需要手动展开!
 */
__global__ void gelu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        // sqrt(2/pi) ≈ 0.79788456
        float a = 0.79788456f * (val + 0.044715f * val * val * val);
        float tanh_a = tanhf(a);  // CUDA 原生支持 tanhf!
        y[idx] = 0.5f * val * (1.0f + tanh_a);
    }
}

// ============================================================
// Element-wise Add: z = x + y
// ============================================================
__global__ void add_kernel(const float* x, const float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

// ============================================================
// 包装函数
// ============================================================
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    int n = x.numel();
    auto y = torch::empty_like(x);
    int block_size = 1024;
    int num_blocks = ceil_div(n, block_size);
    relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor silu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    int n = x.numel();
    auto y = torch::empty_like(x);
    int block_size = 1024;
    int num_blocks = ceil_div(n, block_size);
    silu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    int n = x.numel();
    auto y = torch::empty_like(x);
    int block_size = 1024;
    int num_blocks = ceil_div(n, block_size);
    gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}

torch::Tensor add_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(y.is_cuda() && y.is_contiguous());
    TORCH_CHECK(x.numel() == y.numel(), "输入大小必须一致");
    int n = x.numel();
    auto z = torch::empty_like(x);
    int block_size = 1024;
    int num_blocks = ceil_div(n, block_size);
    add_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), n);
    return z;
}
