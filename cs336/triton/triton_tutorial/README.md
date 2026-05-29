# Triton 从零开始教程

一套循序渐进的动手教程，帮助你从零开始掌握 [Triton](https://github.com/triton-lang/triton) GPU 编程。

## 目录

每一章包含 **教程文档 (`.md`)** + **可运行代码 (`.py`)**，建议先读教程再跑代码。

| 章节 | 教程 | 代码 | 内容 | 预计时间 |
|------|------|------|------|----------|
| 1 | [教程](01_introduction_tutorial.md) | [代码](01_introduction.py) | Hello World, Grid/Block/Thread, tl.load/store | 30 min |
| 2 | [教程](02_elementwise_tutorial.md) | [代码](02_elementwise.py) | ReLU, Silu, GeLU, 性能对比 | 45 min |
| 3 | [教程](03_softmax_tutorial.md) | [代码](03_softmax.py) | Softmax, LayerNorm, 行级聚合 | 45 min |
| 4 | [教程](04_matmul_tutorial.md) | [代码](04_matmul.py) | 矩阵乘法 Tiling, tl.dot | 60 min |
| 5 | [教程](05_flash_attention_tutorial.md) | [代码](05_flash_attention.py) | **FlashAttention Forward** (完整 Triton Kernel) | 90 min |
| 6 | [教程](06_backward_tutorial.md) | [代码](06_backward.py) | FlashAttention Backward, Recomputation | 60 min |
| 7 | [教程](07_reference_tutorial.md) | [代码](07_reference.py) | 速查表, 性能优化, 调试技巧, 学习路径 | 参考 |

## 前置要求

- Python 3.10+
- PyTorch (与你的 CUDA 版本匹配)
- Triton
- GPU (NVIDIA, 建议显存 >= 8GB)

## 运行方式

```bash
cd triton_tutorial

# 逐章: 先读 .md 教程, 再跑 .py 代码
python 01_introduction.py
python 02_elementwise.py
python 03_softmax.py
python 04_matmul.py
python 05_flash_attention.py
python 06_backward.py
python 07_reference.py
```

## 学习建议

1. **不要跳章** — 每一章都建立在前一章的基础上
2. **先读教程再跑代码** — 理解原理后再看实现
3. **修改参数** — 改变 BLOCK_SIZE、序列长度等，观察效果
4. **做练习** — 每章都有动手练习和思考题（教程中标记了 `练习`）
5. **跑测试** — 每章代码末尾都有验证测试

## 配套资料

本教程基于 Stanford CS336 (Language Modeling from Scratch) 课程内容，主要参考:

- Lecture 6 (GPU 高性能编程)
- Assignment 2 (Systems — Triton kernel 实现)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)

## 进阶

完成本教程后，你可以:

1. 完成 [Assignment 2](../assignment2-systems/) 中的 Triton 练习
2. 阅读 [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691) 并实现优化版
3. 参与 vLLM 的 kernel 优化工作
4. 阅读 Triton 源码理解编译器实现
