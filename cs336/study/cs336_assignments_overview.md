# CS336 Assignments Overview

这份文档归纳了 CS336 课程五个 Assignment 的核心内容、技术要点和学习目标，帮助您快速把握课程脉络。

---

## 1. Assignment 1: Basics (基础架构)

> **目标**: 从零构建一个完整的 Transformer 语言模型，理解每一个组件的数学原理和代码实现。

### 🔑 核心任务
1.  **Tokenizer (BPE)**: 实现 Byte-Pair Encoding 分词器，包括训练（从语料学习合并规则）和推理（编码/解码）。
2.  **Attention Mechanism**:
    -   实现朴素的 Scaled Dot-Product Attention。
    -   实现 Multi-Head Attention (MHA)。
    -   实现 **RoPE (Rotary Positional Embedding)**，理解旋转位置编码的数学原理。
3.  **Transformer 组件**:
    -   **RMSNorm**: 实现 Root Mean Square Normalization（比 LayerNorm 更高效）。
    -   **SwiGLU**: 实现带有 SiLU 激活函数的门控前馈网络（LLaMA/Qwen 的标配）。
4.  **完整模型**: 组装 `BasicsTransformerLM`，包括 Embedding, Transformer Block, LM Head。
5.  **训练循环**: 实现 Cross Entropy Loss, AdamW 优化器, Cosine Learning Rate Schedule, Gradient Clipping。

### 💡 技术要点
-   **数值稳定性**: Softmax 和 Cross Entropy 的实现需要使用 LogSumExp trick 防止溢出。
-   **RoPE**: 将位置信息编码为复数旋转，使得模型能自然地处理相对位置。
-   **BPE**: 通过迭代合并最频繁的字节对来构建词表，平衡词表大小和序列长度。

---

## 2. Assignment 2: Systems (系统优化)

> **目标**: 深入理解 LLM 的系统级优化，包括单卡计算效率（FlashAttention）和多卡分布式训练（DDP, ZeRO）。

### 🔑 核心任务
1.  **FlashAttention**:
    -   **PyTorch 版**: 用 PyTorch 原语实现，理解算法逻辑。
    -   **Triton 版**: 用 Triton 编写 GPU Kernel，手动管理 SRAM 和 HBM 之间的数据搬运。
    -   **核心思想**: Tiling (分块) + Online Softmax (在线归一化)，减少 HBM 访问次数。
2.  **Distributed Data Parallel (DDP)**:
    -   实现多卡梯度同步。
    -   **Bucketing**: 将多个参数的梯度打包成桶（Bucket）进行 AllReduce，以重叠计算和通信（Overlap）。
3.  **Sharded Optimizer (ZeRO-1)**:
    -   将优化器状态（Momentum, Variance）切分到不同 GPU 上，减少单卡显存占用。

### 💡 技术要点
-   **IO-Awareness**: 现代 GPU 的瓶颈往往在显存带宽（HBM）而非计算（Tensor Cores）。FlashAttention 通过减少 HBM 读写实现加速。
-   **Communication-Computation Overlap**: 在反向传播计算梯度的同时，异步发送已经计算好的梯度进行同步。
-   **Memory Optimization**: ZeRO 系列通过切分状态，使得显存占用与 GPU 数量成反比。

---

## 3. Assignment 3: Scaling (扩展定律)

> **目标**: 理解 Scaling Laws（扩展定律），学会如何在训练前预测模型性能，并根据算力预算进行最优分配。

### 🔑 核心任务
1.  **FLOPs 计算**: 精确计算 Transformer模型在训练和推理时的浮点运算量。
2.  **Scaling Laws 拟合**:
    -   使用实验数据（Loss vs 参数量 N vs 数据量 D）拟合 Chinchilla Scaling Law 公式。
    -   公式形式：$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$。
3.  **IsoFLOP 分析**:
    -   绘制固定 FLOPs 下的 Loss 曲线，找到最优的 (模型大小, 数据量) 组合。
4.  **算力分配**: 给定计算预算（如 $10^{24}$ FLOPs），决定应该训练多大的模型以及用多少数据。

### 💡 技术要点
-   **Chinchilla Optimal**: 计算预算增加时，模型参数量 ($N$) 和数据量 ($D$) 应该等比例增加（各占一半）。
-   **Compute-Bound**: 大模型训练通常是计算受限的。
-   **Over-training**: 在推理成本极其敏感的场景（如 LLaMA 3），可能会训练比 Chinchilla 最优小得多的模型，但用更多的数据（Inference-optimal）。

---

## 4. Assignment 4: Data (数据处理)

> **目标**: 构建从原始网页到高质量预训练语料的完整处理 Pipeline。

### 🔑 核心任务
1.  **文本提取**: 从原始 HTML 中提取干净的文本内容。
2.  **PII 脱敏**: 使用正则表达式识别并屏蔽个人隐私信息（邮箱、电话、IP）。
3.  **质量过滤**:
    -   **Gopher 规则**: 基于文本长度、单词长度、符号比例等启发式规则过滤低质量文本。
    -   **分类器**: 训练/使用模型识别有害内容（Toxicity, NSFW）。
4.  **去重 (Deduplication)**:
    -   **Exact Dedup**: 精确行级去重。
    -   **Fuzzy Dedup (MinHash)**: 使用 MinHash + LSH (Locality Sensitive Hashing) 识别近似重复的文档（如转载文章）。

### 💡 技术要点
-   **Data Quality is King**: 数据质量对模型性能的影响往往大于模型架构的改进。
-   **MinHash + LSH**: 大规模数据去重的标准算法，能在 $O(1)$ 时间内找到相似文档。
-   **Safety**: 在预训练阶段就过滤掉有害内容，比后期对齐更有效。

---

## 5. Assignment 5: Alignment (对齐)

> **目标**: 使用 SFT 和 RLHF (GRPO/DPO) 技术将预训练模型对齐到人类意图。

### 🔑 核心任务
1.  **SFT (Supervised Fine-Tuning)**:
    -   构建 Instruction Tuning 数据集。
    -   实现带 Mask 的 Loss（只计算 Response 部分的 Loss）。
2.  **GRPO (Group Relative Policy Optimization)**:
    -   复现 DeepSeek-R1 的核心算法。
    -   为每个 Prompt 生成一组回复 (Group)。
    -   使用组内相对优势 (Group Relative Advantage) 作为训练信号，无需训练额外的 Critic 模型。
3.  **DPO (Direct Preference Optimization)**:
    -   实现 DPO Loss，直接使用偏好数据对 $(x, y_w, y_l)$ 优化策略，无需 Reward Model。
4.  **评估**:
    -   实现 MMLU（知识能力）和 GSM8K（数学推理能力）的自动化评估。

### 💡 技术要点
-   **SFT vs RLHF**: SFT 学习格式和基础知识，RLHF (GRPO/DPO) 学习偏好和推理逻辑。
-   **GRPO**: 相比 PPO 更简单高效，不需要值函数网络，适合推理任务（DeepSeek-R1）。
-   **DPO**: 数学上等价于 RLHF，但通过转化为分类问题避免了复杂的强化学习训练过程。

---

这份概览涵盖了从模型定义、系统优化、Scaling分析、数据处理到后训练对齐的全过程，是理解 LLM 全栈技术的绝佳路线图。
