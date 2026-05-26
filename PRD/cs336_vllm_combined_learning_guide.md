# CS336 × vLLM 联合学习指南

## 核心思路

**CS336** 教你从零手写大模型全栈（训练侧），**vLLM** 展示工业级推理优化。两者的交汇点是 **系统级优化**：在 CS336 里懂原理、写基础版，去 vLLM 里看工业级实现、学极致优化。

```
CS336 (理论 + 造轮子)          vLLM (工业级推理落地)
━━━━━━━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━━━━━━━
Assignment1: 基础模型    ──→   vllm/model_executor/: 模型加载与执行
Assignment2: 系统优化    ──→   csrc/*.cu: 生产级 CUDA Kernel
Assignment3: Scaling     ──→   vllm/config/: 模型配置与资源计算
Lecture 06: GPU Kernel   ──→   csrc/layernorm_kernels.cu, pos_encoding_kernels.cu
Lecture 08: 分布式       ──→   vllm/distributed/, csrc/custom_all_reduce.cu
Lecture 10: 推理优化     ──→   csrc/attention/, vllm/v1/core/, vllm/v1/spec_decode/
```

---

## 模块一：Transformer 基础与模型实现

### CS336 学习内容
| 资源 | 主题 | 关键知识点 |
|------|------|-----------|
| Lecture 01 | 课程总览 | Transformer 架构变体、SwiGLU、RoPE、RMSNorm、GQA/MLA |
| Lecture 02 | 训练原语 | 张量操作、FLOPs 计算、内存计算、混合精度训练 (AMP) |
| Assignment 1 | 从零实现 | BPE Tokenizer、Transformer、CrossEntropy、AdamW、训练循环 |

### vLLM 对标阅读
| vLLM 路径 | 对标内容 |
|-----------|---------|
| `vllm/model_executor/models/` | 各模型（Llama, Qwen, DeepSeek）的工业级实现 |
| `vllm/transformers_utils/` | Tokenizer 的工业级封装 |

### 学习任务
- [ ] 完成 Assignment 1，手写 Transformer + BPE + AdamW
- [ ] 打开 `vllm/model_executor/models/` 中任一模型（如 Llama），对比你的实现：观察 GQA、RoPE、RMSNorm 的写法差异
- [ ] **面试关联**：能手写 Forward/Backward、一层 MLP 实现

---

## 模块二：GPU Kernel 编写与优化 ⭐ (最高优先级)

### CS336 学习内容
| 资源 | 主题 | 关键知识点 |
|------|------|-----------|
| Lecture 06 | GPU 编程 | 硬件架构 (SM/L1/L2/HBM)、执行模型 (Thread/Block/Grid)、Benchmarking/Profiling、Kernel Fusion、CUDA/Triton 编写 GeLU、PTX 指令 |
| Assignment 2 | 系统优化 | 实现 Fused RMSNorm (Triton)、FlashAttention 基础 |
| `gelu.cu` | CUDA 示例 | 完整的 elementwise kernel 示例 |

### vLLM 对标阅读
| vLLM 路径 | 对标内容 | 学什么 |
|-----------|---------|--------|
| `csrc/layernorm_kernels.cu` | RMSNorm CUDA | Warp-level Reduction + 向量化访存 |
| `csrc/pos_encoding_kernels.cu` | RoPE CUDA | Elementwise kernel + 位置编码实现 |
| `csrc/activation_kernels.cu` | 激活函数 CUDA | Fused SiLU/GeLU 等 |
| `csrc/attention/attention_kernels.cuh` | Paged Attention | 核心：Block-level Reduction + Shared Memory 管理 |
| `csrc/attention/paged_attention_v1.cu` | PagedAttn V1 | 基础版 Paged Attention |
| `csrc/attention/paged_attention_v2.cu` | PagedAttn V2 | Split-K 优化版本 |
| `csrc/sampler.cu` | Top-K Sampling | Top-K Selection Kernel |

### 学习任务
- [ ] 跟着 Lecture 06 手写 GeLU CUDA kernel，理解 `blockIdx`, `threadIdx` 的 Index 计算
- [ ] 完成 Assignment 2 的 Fused RMSNorm Triton kernel
- [ ] 打开 `csrc/layernorm_kernels.cu`，对比你的 Triton 版本，回答：
  - vLLM 如何做 Warp-level Reduction？用了哪些 `__shfl_xor_sync`？
  - 为什么用 `float4` 向量化读取？性能差多少？
- [ ] 研读 `csrc/attention/attention_kernels.cuh` 的核心循环，理解：
  - 如何分块处理 KV Cache（对应面试中的 Flash Attention Softmax 分块）
  - Shared Memory 如何存储 K/V 并避免 Bank Conflict
- [ ] **面试关联**：Reduction (高频)、Elementwise (高频)、SGEMM (中频)、Top-K (低频)

---

## 模块三：注意力机制与内存管理 ⭐

### CS336 学习内容
| 资源 | 主题 | 关键知识点 |
|------|------|-----------|
| Lecture 10 | 推理优化 | KV Cache 原理、Prefill vs Decode、Compute-bound vs Memory-bound 分析、GQA/MLA/CLA、量化 (int8/fp8/AWQ)、Speculative Decoding |
| Assignment 2 | FlashAttention | FlashAttention 的分块计算原理 |
| `study/cs336_assignment2_flash_attention_details.md` | FA 细节 | 你已有的 FlashAttention 笔记 |

### vLLM 对标阅读
| vLLM 路径 | 对标内容 | 学什么 |
|-----------|---------|--------|
| `csrc/attention/paged_attention_v1.cu` | PagedAttention | 将 OS 虚拟内存分页思想引入 KV Cache |
| `csrc/attention/mla/` | MLA Attention | DeepSeek 的 Multi-head Latent Attention |
| `csrc/cache_kernels.cu` | Cache 管理 | KV Cache 的 GPU 端 copy/reshape |
| `vllm/v1/core/` | 调度器核心 | Block 分配、Continuous Batching 调度 |
| `vllm/v1/spec_decode/` | 推测解码 | Speculative Decoding 的工业级实现 |
| `vllm/vllm_flash_attn/` | FlashAttention | vLLM 定制的 FlashAttention 实现 |

### 学习任务
- [ ] 精读 Lecture 10，手推 MLP 层和 Attention 层的 Arithmetic Intensity 公式
- [ ] 理解为什么 Prefill 是 Compute-bound，Decode 是 Memory-bound
- [ ] 完成 Assignment 2 中的 FlashAttention 部分
- [ ] 研读 `csrc/attention/paged_attention_v1.cu`：
  - 与标准 Attention 对比，PagedAttention 是怎么通过 block_table 索引离散 KV 的？
  - 与 `paged_attention_v2.cu` 对比，Split-K 优化了什么场景？
- [ ] 浏览 `vllm/v1/core/` 理解 Continuous Batching 的调度逻辑
- [ ] 浏览 `vllm/v1/spec_decode/` 对照 Lecture 10 中的 Speculative Sampling 算法
- [ ] **面试关联**：Flash Attention 三版本 (高频)、Compute vs Memory bound (高频)

---

## 模块四：分布式计算与通信

### CS336 学习内容
| 资源 | 主题 | 关键知识点 |
|------|------|-----------|
| Lecture 08 | 分布式训练 | 集合通信原语 (Broadcast/Scatter/Gather/Reduce/All-Reduce)、NCCL、NVLink 拓扑、Data/Tensor/Pipeline Parallelism 裸实现 |
| Assignment 2 | DDP | 实现分布式数据并行训练 + Optimizer State Sharding |
| Assignment 3 | Scaling Laws | Chinchilla 最优、超参预测 |

### vLLM 对标阅读
| vLLM 路径 | 对标内容 | 学什么 |
|-----------|---------|--------|
| `vllm/distributed/parallel_state.py` | 并行状态管理 | TP/PP 的 Process Group 划分 |
| `vllm/distributed/communication_op.py` | 通信原语封装 | All-Reduce/All-Gather 的 Python 封装 |
| `csrc/custom_all_reduce.cu` | 自定义 All-Reduce | vLLM 极致优化的集合通信 Kernel |
| `vllm/distributed/device_communicators/` | 设备通信器 | NCCL/Custom 后端选择逻辑 |

### 学习任务
- [ ] 跟着 Lecture 08 理解 All-Reduce = Reduce-Scatter + All-Gather
- [ ] 完成 Assignment 2 的 DDP 和 Optimizer Sharding 实现
- [ ] 完成 Assignment 3，拟合 Scaling Law，做超参数预测
- [ ] 打开 `vllm/distributed/parallel_state.py`，理解：
  - 推理场景下 Tensor Parallelism 如何划分 Process Group
  - 与训练场景 (CS336 Lecture 08) 的 DP/TP/PP 有何异同
- [ ] 研读 `csrc/custom_all_reduce.cu`，理解 vLLM 为何要自己写 All-Reduce 而非直接用 NCCL
- [ ] **面试关联**：Collective Communication (中频)、Megatron 张量切分 (中频)、MHA 并行化

---

## 模块五：数据处理与对齐

### CS336 学习内容
| 资源 | 主题 | 关键知识点 |
|------|------|-----------|
| Lecture 13 | 数据集总览 | CommonCrawl、The Pile、FineWeb、DCLM；数据处理 Pipeline |
| Lecture 14 | 数据处理 | KenLM/fastText 过滤、Bloom Filter 去重、MinHash + LSH 近似去重 |
| Lecture 17 | GRPO | Policy Gradient、Baseline/Advantage、GRPO 算法 |
| Assignment 4 | 数据工程 | HTML→Text、质量分类器、MinHash 去重 |
| Assignment 5 | 对齐 | SFT、DPO、GRPO 实现 |

### vLLM 对标关联
此模块主要偏训练侧，vLLM 的直接对标较少，但以下路径仍有价值：

| vLLM 路径 | 关联点 |
|-----------|--------|
| `vllm/v1/sample/` | Sampling 策略（Top-K/Top-P/Temperature），理解推理端如何使用这些策略 |
| `vllm/beam_search.py` | Beam Search，对标面经中的 Top-K Selection |

### 学习任务
- [ ] 完成 Assignment 4，实现数据清洗全流程
- [ ] 完成 Assignment 5，实现 SFT + DPO + GRPO
- [ ] **面试关联**：拓扑排序 (用于反向传播图)、位运算优化

---

## 模块六：评估与工程实践

### CS336 学习内容
| 资源 | 主题 |
|------|------|
| Lecture 12 | 模型评估：MMLU/GPQA/SWEBench/Chatbot Arena、Safety、Train-test overlap |

### vLLM 工程实践
| vLLM 路径 | 学什么 |
|-----------|--------|
| `benchmarks/` | 性能基准测试：吞吐量、延迟测量方法 |
| `vllm/profiler/` | GPU Profiling 工具封装 |
| `vllm/config/` | 模型/缓存/并行配置的设计模式 |
| `CMakeLists.txt` + `setup.py` | C++/Python 混合项目的构建系统 |

### 学习任务
- [ ] 浏览 `benchmarks/` 理解工业级推理性能测量方法
- [ ] 研读 `CMakeLists.txt`，理解 CUDA 项目的 CMake 构建
- [ ] **面试关联**：CI/CD、CMake、Docker/K8s 基础

---

## 推荐学习顺序

```
Week 1-2: 模块一 (Transformer 基础) ← Lecture 01-02 + Assignment 1
    │
Week 3-4: 模块二 (GPU Kernel) ← Lecture 06 + Assignment 2 Part 1
    │      同时开始阅读 csrc/layernorm_kernels.cu
    │
Week 5-6: 模块三 (注意力 & 内存) ← Lecture 10 + Assignment 2 Part 2
    │      同时阅读 csrc/attention/paged_attention_v1.cu
    │
Week 7-8: 模块四 (分布式) ← Lecture 08 + Assignment 2 Part 3 + Assignment 3
    │      同时阅读 vllm/distributed/ + csrc/custom_all_reduce.cu
    │
Week 9-10: 模块五 (数据 & 对齐) ← Lecture 13-14, 17 + Assignment 4-5
    │
Week 11+: 模块六 (工程实践) + 查漏补缺 + 刷 LeetGPU
```

---

## 每日学习模板

1. **上午：CS336 Lecture / Assignment**（概念输入 + 动手实现）
2. **下午：vLLM 源码对标阅读**（看工业级代码是怎么写的）
3. **晚上：面试题练习**（LeetGPU / 手撕 Kernel / LeetCode）

### 阅读 vLLM 源码时的两个核心问题

> **Q1**: vLLM 的代码为什么要这么写？
> （是为了避免 Bank Conflict？减少 Register Spill？还是为了更好的 Warp Occupancy？）

> **Q2**: 如果面试官让我把 CS336 作业代码优化到 vLLM 水平，我讲得出思路吗？

---

## 面经知识点 → CS336/vLLM 速查表

| 面经考点 | CS336 资源 | vLLM 源码 |
|---------|-----------|----------|
| GPU 架构 (SM/Warp/Block) | Lecture 06 | — |
| Roofline / Compute vs Memory bound | Lecture 06, 10 | — |
| Coalesced Read / Bank Conflict | Lecture 06 | `csrc/attention/attention_kernels.cuh` |
| Kernel Fusion | Lecture 06 (GeLU 对比) | `csrc/activation_kernels.cu` |
| Reduction Kernel | Lecture 06 (Softmax profiling) | `csrc/layernorm_kernels.cu` |
| FlashAttention | Assignment 2 + study notes | `vllm/vllm_flash_attn/` |
| Transformer / Attention | Lecture 01-02, Assignment 1 | `vllm/model_executor/models/` |
| KV Cache / PagedAttention | Lecture 10 | `csrc/attention/paged_attention_*.cu` |
| GQA / MLA | Lecture 10 | `csrc/attention/mla/` |
| Speculative Decoding | Lecture 10 | `vllm/v1/spec_decode/` |
| Collective Communication | Lecture 08 | `csrc/custom_all_reduce.cu` |
| DP / TP / PP | Lecture 08 | `vllm/distributed/parallel_state.py` |
| Scaling Laws | Lecture 02, Assignment 3 | — |
| 量化 (int8/fp8/AWQ) | Lecture 10 | `csrc/quantization/` |
| Top-K Selection | — | `csrc/sampler.cu` |
| 数据过滤 / 去重 | Lecture 13-14, Assignment 4 | — |
| SFT / DPO / GRPO | Lecture 17, Assignment 5 | — |
| CMake / 编译链接 | — | `CMakeLists.txt` |
