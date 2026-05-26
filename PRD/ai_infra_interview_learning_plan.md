# AI Infra & GPU Kernel 研发岗面试复习与学习规划 (PRD)

## 文档概述
本文档基于一份北美 AI Infra / GPU Kernel 开发岗位的面经总结，梳理了面试中的高频考点和重难点，并将其转化为具体的学习任务清单（TODO list）与需求规划。此文档既作为个人的复习指南，也作为学习进度的执行需求说明书。

## 阶段一：基础知识巩固 (Foundation & Core Concepts)

### 1.1 GPU 架构与编程体系 (优先级：高)
- [ ] **硬件与软件模型**: 深入理解 GPU 硬件架构与软件编程模型 (Thread, Block, Grid, Warp)。
- [ ] **新特性追踪**: 学习 Nvidia 新特性：TMA (Tensor Memory Accelerator), Cutlass 库, CuTe DSL。
- [ ] **性能分析理论**: 掌握 Roofline Model, Brent's Theorem，准确区分 Compute-bound 和 Memory-bound。
- [ ] **分析工具**: 熟悉 Profiler 工具：Nsight Compute / Nsight System 的基本概念与指标解读。
- [ ] **基础优化方法**:
  - [ ] Coalesced Read (合并访存) 的原理与实现。
  - [ ] Shared Memory Bank Conflict 避免技巧 (Padding / Swizzling)。
  - [ ] 向量化内存读取 (Vectorized memory access, 比如 float4)。
  - [ ] 减少 Register Spill 的策略。
  - [ ] Warp Specialization 概念。

### 1.2 机器学习与深度学习底层原理 (优先级：高)
- [ ] **NLP & 大模型**: Transformer 架构，RNN 基础，Attention 机制细节 (非常重要)。
- [ ] **Flash Attention**: 深入剖析 Flash Attention 的三个版本演进细节 (v1, v2, v3)。
- [ ] **实践编码**: 能够手写前向/反向传播 (Forward/Backward)，实现简单的一层 MLP。
  - *推荐资源：Andrew Karpathy 的系列视频教程，Deep ML 刷题练手。*
- [ ] **CV 基础 (低频)**: CNN 时空复杂度计算，ResNet, ViT，理解 U-Net 中的 Up-scale 机制。

### 1.3 分布式计算框架 (优先级：中)
- [ ] **理论基础**: 理解 Strong Scaling 与 Weak Scaling 的区别。
- [ ] **集合通信**: 掌握常用 Collective Communication 原理及语句 (All-Reduce, All-Gather, Broadcast 等)。
- [ ] **模型并行策略**: 掌握 Pipeline Parallelism (PP), Model Parallelism (MP), Fully Sharded Data Parallel (FSDP)。
- [ ] **张量切片**: 了解 Megatron 的张量切分 (先行为切，再列切)。
- [ ] **思考题**: Multi-head Attention 如何实现并行？
- [ ] **进阶探索**: 了解 RDMA, MPI+GPU, 以及当下较火的 PGAS/SHMEM 通信模式。

### 1.4 系统架构与基础设施 (优先级：中)
- [ ] **C/C++ 进阶**: const pointer vs pointer to const, 虚函数与继承机制。
- [ ] **设计模式**: 重点掌握 Strategy Pattern (策略模式) 和 Abstract Factory Pattern (抽象工厂模式)。
- [ ] **编译与链接**: CMake 编写，动态链接与静态链接的区别。
- [ ] **CI/CD 基础**: Pipeline 基础理论，如何处理 Fault？如何执行 Rollback？
- [ ] **容器与部署**: Docker/Kubernetes 基础概念 (如 Image 的上传与管理)。
- [ ] **包管理**: Python env 和 package management。
- [ ] **选修**: C++ 模板元编程 (Template Metaprogramming)。

### 1.5 操作系统与计算机网络基础 (优先级：低)
- [ ] **内存系统**: Write-through vs Write-back，Memory Fence，Heap vs Stack。
- [ ] **并发编程**: 异步编程及 Lock-free Programming 基础。
- [ ] **系统底层**: 内核态与用户态切换机制，Zero Copy 技术。
- [ ] **网络基础**: OSI 五层网络模型，TCP 三次握手过程。

---

## 阶段二：GPU Kernel 手撕实战 (Kernel Coding Practice)

此阶段为核心考察点，建议前往 **LeetGPU** 刷题，熟悉 CUDA 乃至 Triton / CuTe 的编写。

### 2.1 核心必考 Kernel
- [ ] **Reduction (高频)**: 掌握 Warp-level / Block-level Reduction 的最优写法。理解 Mark Harris 的经典实现。
  - *延伸*: 掌握 Softmax, Layer Norm, Batch Norm 实现，以及由此引申的 Flash Attention 中 Softmax 分块计算的实现细节。
- [ ] **Elementwise (高频)**: 熟练编写 Sigmoid 等 elementwise 操作。
  - *延伸*: 能够处理 Warp Divergence。学会使用 Mask 来消除分支预测 (例如 `check = c < N; v += check * a + (1 - check) * b;`)。

### 2.2 进阶考点 Kernel
- [ ] **SGEMM (中频)**: 掌握 Block Tiling 的 Index 计算和 Shared Memory 缓存。
  - *延伸*: 能够回答 Split-K 的优化思想和后续问题。
- [ ] **Scan (中频)**: 掌握 GPU Gems 3 (Mark Harris) 提到的 Prefix Sum。
  - *延伸*: 实现 Stream Compaction (常用于 DL 中处理 Sparse Data)，了解 Radix Sort (基数排序) 的口述思路。
- [ ] **Top K (低频)**: 实现 LLM Beam Search 中的 Top-K selection。
  - *Naive 方法*: Warp Shuffle 选最大值后 in-place 替换为 `-FLOATMAX`。
  - *优化方法*: Bitonic Sort 的写法 (参考 Tri Dao 的 CuTe 实现)。
- [ ] **Transpose (低频)**: 利用 Shared memory 作为中间缓冲层来实现 Coalesced data visit (合并访存)。

---

## 阶段三：算法题与系统设计 (LeetCode & System Design)

### 3.1 数据结构与算法 (目标水平：LeetCode 1200-1700 周赛难度)
- [ ] **重点算法**: 二分查找，双指针，贪心算法。
- [ ] **拓扑排序 (Topology Sorting)**: 极为关键，常用于 Back Propagation 图构建或编译器 Instruction Order 处理。
- [ ] **位运算 (高频)**: C++ 底层性能优化必备。
  - 判断奇偶: `a & 1`
  - 向量化/2的幂次运算技巧: `div = x >> 2` (对4求商取整), `mod = x & (4 - 1)` (对4求模)。

### 3.2 系统设计
- [ ] 设计实现简单的 **Load Balancer** (例如 Round Robin 轮询机制)。
- [ ] 设计基于**异步读写**的 Double Buffering (双缓冲) 机制。
- [ ] 了解简易的**推荐系统**设计和基本链路。

---

## 阶段四：极限挑战 / Bar Raiser 专属储备

此类题目通常由 Skip Manager 在定级面 (Bar Raiser) 提出，难度极大。旨在考察对系统底层的极致理解和抗压能力。做不出无伤大雅，但做出来能极大提升定级 (Level/Package)。
- [ ] **挑战 1**: 熟悉 Flash Attention V1 的 CUDA 核心实现，尝试能用 Triton 现场写一个简化版本。
- [ ] **挑战 2**: 掌握二维矩阵乘 (2D GEMM) 的 CUDA 现场手撕能力，探索三维矩阵乘 (3D GEMM) 的实现思路。
- [ ] **挑战 3**: 学习如何使用 PTX (Parallel Thread Execution) 指令集来做 GEMM 极致优化。
- [ ] **挑战 4**: 保持手感，定期复习 Hard 级别的动态规划 (DP) 算法题。
