/*
 * PyTorch C++ Extension Bindings
 * ===============================
 *
 * 使用 pybind11 (通过 torch/extension.h) 将 CUDA kernel 包装成 Python 函数。
 *
 * 编译后，你可以在 Python 中这样使用:
 *   import cuda_tutorial
 *   y = cuda_tutorial.hello(x)
 *   y = cuda_tutorial.relu(x)
 *   ...
 */

#include <torch/extension.h>

// ---- Forward declarations (定义在各 .cu 文件中) ----

// 01_hello.cu
torch::Tensor hello_forward(torch::Tensor x);

// 02_elementwise.cu
torch::Tensor relu_forward(torch::Tensor x);
torch::Tensor silu_forward(torch::Tensor x);
torch::Tensor gelu_forward(torch::Tensor x);
torch::Tensor add_forward(torch::Tensor x, torch::Tensor y);

// 03_softmax.cu
torch::Tensor softmax_forward(torch::Tensor x);
torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor gamma,
                                 torch::Tensor beta, double eps);

// 04_matmul.cu
torch::Tensor matmul_forward(torch::Tensor A, torch::Tensor B);

// 05_flash_attention.cu
torch::Tensor flash_attn_forward(torch::Tensor Q, torch::Tensor K,
                                  torch::Tensor V, bool is_causal);

// ---- Python Module ----
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA C++ 教程算子 — 对应 Triton Tutorial";

    // 第一章: Hello (add 1)
    m.def("hello", &hello_forward, "向量加 1 (对应 01_introduction)");

    // 第二章: Elementwise
    m.def("relu", &relu_forward, "ReLU 激活函数");
    m.def("silu", &silu_forward, "Silu/Swish 激活函数");
    m.def("gelu", &gelu_forward, "GeLU 激活函数");
    m.def("add", &add_forward, "Element-wise 加法");

    // 第三章: Row-wise
    m.def("softmax", &softmax_forward, "行级 Softmax");
    m.def("layernorm", &layernorm_forward, "Layer Normalization",
          py::arg("x"), py::arg("gamma"), py::arg("beta"),
          py::arg("eps") = 1e-5);

    // 第四章: MatMul
    m.def("matmul", &matmul_forward, "Tiled 矩阵乘法");

    // 第五章: FlashAttention
    m.def("flash_attn", &flash_attn_forward, "FlashAttention Forward",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("is_causal") = false);
}
