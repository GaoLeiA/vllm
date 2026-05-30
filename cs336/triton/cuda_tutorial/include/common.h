#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

// ============================================================
// CUDA 错误检查宏
// ============================================================
// 用法: CUDA_CHECK(cudaMemcpy(...));
// 如果 CUDA 调用失败，会打印错误信息并退出程序
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// 工具函数
// ============================================================

// 向上取整除法: ceil(a / b)
// 等价于 Triton 的 triton.cdiv(a, b)
inline __host__ __device__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// 计算 >= n 的最小 2 的幂
// 等价于 Triton 的 triton.next_power_of_2(n)
inline int next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}
