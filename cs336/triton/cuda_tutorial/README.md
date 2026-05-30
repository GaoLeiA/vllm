# CUDA C++ 教程 — 对应 Triton Tutorial

本目录包含与 `triton_tutorial/` 一一对应的 CUDA C++ 算子实现。

## 目录结构

```
cuda_tutorial/
├── include/common.h              # 共用宏和工具函数
├── kernels/
│   ├── 01_hello.cu               # 向量加 1
│   ├── 02_elementwise.cu         # ReLU, Silu, GeLU, Add
│   ├── 03_softmax.cu             # Softmax, LayerNorm (shared memory reduction)
│   ├── 04_matmul.cu              # Tiled MatMul (shared memory tiling)
│   └── 05_flash_attention.cu     # FlashAttention Forward (online softmax)
├── bindings.cpp                  # pybind11 Python 绑定
├── setup.py                      # 编译脚本
├── 01_test_hello.py              # 测试脚本
├── 02_test_elementwise.py
├── 03_test_softmax.py
├── 04_test_matmul.py
└── 05_test_flash_attention.py
```

## 编译 & 运行

```bash
conda activate pyre
cd triton/cuda_tutorial

# 编译安装
pip install -e .

# 运行测试
python 01_test_hello.py
python 02_test_elementwise.py
python 03_test_softmax.py
python 04_test_matmul.py
python 05_test_flash_attention.py
```

## Triton vs CUDA 对比

| 概念 | Triton | CUDA |
|------|--------|------|
| 线程管理 | 编译器自动管理 Thread | 手动管理 Grid/Block/Thread |
| Shared Memory | 编译器自动使用 | 手动 `__shared__` + `__syncthreads()` |
| Reduction | `tl.max()`, `tl.sum()` 一行搞定 | 手动写树形归约 (~10 行) |
| Mask | `tl.load(ptr, mask=mask)` | `if (idx < n)` |
| 矩阵乘法 | `tl.dot(a, b)` | 手动 shared memory tiling |
| 编译 | JIT (运行时编译) | 预编译 (nvcc) |
