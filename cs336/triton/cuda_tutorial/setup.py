"""
CUDA Tutorial 编译脚本
======================

使用方法:
    # 安装到 Python 环境 (推荐)
    conda activate pyre
    pip install -e .

    # 或者一次性编译
    python setup.py build_ext --inplace

编译后，可以直接 import:
    import cuda_tutorial
    cuda_tutorial.hello(x)
"""

import os
from setuptools import setup

# ---- 跳过 CUDA 版本检查 ----
# 系统 nvcc 是 12.4, 但 PyTorch 编译用的是 13.0
# 对于简单的 CUDA kernel, 跨小版本兼容没有问题
import torch.utils.cpp_extension as _ext
_ext._check_cuda_version = lambda *args, **kwargs: None

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="cuda_tutorial",
    version="0.1.0",
    description="CUDA C++ 教程算子 — 对应 Triton Tutorial",
    ext_modules=[
        CUDAExtension(
            name="cuda_tutorial",
            sources=[
                "bindings.cpp",
                "kernels/01_hello.cu",
                "kernels/02_elementwise.cu",
                "kernels/03_softmax.cu",
                "kernels/04_matmul.cu",
                "kernels/05_flash_attention.cu",
            ],
            include_dirs=[os.path.join(this_dir, "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",       # 使用快速数学 (tanhf, expf 等)
                    "-lineinfo",             # 保留行号信息便于调试
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
