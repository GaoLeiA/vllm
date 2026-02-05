# CS336 学习指南：从基础到 vLLM 实践

## 📚 课程概览

**CS336: Language Modeling from Scratch** 是斯坦福大学的一门深度学习课程，专注于从零开始构建语言模型。本课程涵盖了从基础实现到产品级系统优化的完整知识链。

### 课程与 vLLM 的关系

vLLM 是一个高性能的 LLM 推理和服务框架，实现了许多 CS336 中讨论的高级优化技术。学习 CS336 可以帮助你：

1. **理解 vLLM 的底层原理** - CS336 讲解的技术如 FlashAttention、分布式训练等都在 vLLM 中有实际应用
2. **从头实现核心算法** - 通过作业动手实现，加深对 vLLM 源码的理解
3. **建立系统性思维** - 理解从训练到推理的完整优化链路

### 推荐学习路径

```
CS336 Assignment 1 (基础) → CS336 Assignment 2 (系统优化) 
                                    ↓
                          理解 vLLM 源码架构
                                    ↓
CS336 Assignment 3 (Scaling Laws) ← 规模化训练原理
                                    ↓
CS336 Assignment 4 (数据) → CS336 Assignment 5 (对齐)
```

---

## 📋 Assignment 详细内容

### Assignment 1: Basics（基础）
**主题**: 从零构建 Transformer 语言模型

#### 核心实现内容

| 模块 | 描述 | vLLM 对应 |
|------|------|-----------|
| **Linear/Embedding** | 实现基础神经网络层 | `vllm/model_executor/layers/` |
| **RMSNorm** | 实现 Root Mean Square Normalization | 模型归一化层 |
| **SiLU/SwiGLU** | 激活函数和前馈网络 | FFN 层实现 |
| **Scaled Dot-Product Attention** | 注意力机制基础 | 理解 PagedAttention 的前置知识 |
| **Multi-Head Self-Attention** | 多头注意力 + RoPE 位置编码 | vLLM 的 attention 实现 |
| **Transformer Block** | 完整的 Transformer 层 | 模型架构基础 |
| **Transformer LM** | 完整的语言模型 | 理解如何加载模型权重 |
| **BPE Tokenizer** | 字节对编码分词器训练 | 理解 tokenizer 工作原理 |
| **AdamW Optimizer** | 优化器实现 | 训练相关 |
| **Cosine LR Schedule** | 学习率调度 | 训练相关 |
| **Checkpoint Save/Load** | 模型序列化 | 权重加载机制 |

#### 数据集
- **TinyStories**: 用于训练小规模模型的故事数据集
- **OpenWebText (sample)**: 网页文本样本

#### 关键代码位置
```
cs336/assignment1-basics/
├── cs336_basics/          # 你需要实现的模块
├── tests/adapters.py      # 测试适配器（定义了所有需要实现的接口）
└── tests/test_*.py        # 单元测试
```

---

### Assignment 2: Systems（系统优化）
**主题**: 高性能训练系统实现

#### 核心实现内容

| 模块 | 描述 | vLLM 对应 |
|------|------|-----------|
| **FlashAttention (PyTorch)** | 用纯 PyTorch 实现 FlashAttention | 理解内存优化原理 |
| **FlashAttention (Triton)** | 使用 Triton 编写自定义 CUDA kernel | vLLM 的 Triton kernels |
| **DDP (Individual Parameters)** | 逐参数分布式数据并行 | 多 GPU 训练基础 |
| **DDP (Bucketed)** | 分桶 DDP，优化通信开销 | 理解梯度同步机制 |
| **Sharded Optimizer** | 优化器状态分片（类似 ZeRO-1） | 大模型训练技术 |

#### 关键技术点

**FlashAttention 算法核心**:
```python
# 核心思想：分块计算 + Online Softmax
# 避免存储 O(N²) 的注意力矩阵
for i in range(0, N, block_size):      # 遍历 Q 块
    for j in range(0, N, block_size):  # 遍历 K, V 块
        # 1. 计算局部注意力分数
        # 2. 使用 Online Softmax 更新全局状态
        # 3. 累积输出
```

**分布式训练**:
- **Ring-AllReduce**: 高效的梯度聚合
- **Bucket 策略**: 将小梯度打包，提高通信效率
- **通信-计算重叠**: 异步通信隐藏延迟

#### 与 vLLM 的深度关联
- vLLM 使用 FlashAttention 进行高效推理
- 理解 Triton kernel 编写有助于理解 vLLM 的 `csrc/` 目录

---

### Assignment 3: Scaling（规模化）
**主题**: Scaling Laws（规模法则）研究

#### 核心内容

| 主题 | 描述 |
|------|------|
| **Chinchilla Scaling Laws** | 模型大小 vs 数据量 vs 计算量的最优配比 |
| **实验设计** | 如何设计实验验证 Scaling Laws |
| **Compute-Optimal Training** | 给定计算预算，如何分配模型大小和训练数据 |

#### 关键公式

**Kaplan (OpenAI) Scaling Law**:
$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

**Chinchilla Scaling Law**:
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中:
- $N$ = 模型参数量
- $D$ = 训练数据量 (tokens)
- $L$ = 损失
- $E$ = 不可约损失

#### 实践意义
- 理解为什么 LLaMA 系列在给定规模下表现优秀
- 帮助规划训练资源分配

---

### Assignment 4: Data（数据处理）
**主题**: 训练数据质量与处理

#### 核心实现内容

| 模块 | 描述 | 技术细节 |
|------|------|----------|
| **HTML 文本提取** | 从网页中提取纯文本 | 使用 trafilatura 等工具 |
| **语言识别** | 识别文本语言 | fastText 语言检测 |
| **PII 脱敏** | 个人信息保护 | 邮箱、电话、IP 地址掩码 |
| **NSFW/毒性检测** | 有害内容过滤 | 分类器实现 |
| **质量过滤** | 数据质量评估 | Gopher 质量过滤规则 |
| **精确去重** | 行级别精确匹配 | 哈希去重 |
| **MinHash 去重** | 近似文档去重 | LSH (局部敏感哈希) |

#### Gopher 质量过滤规则示例
```python
def gopher_quality_filter(text: str) -> bool:
    """
    Based on DeepMind's Gopher paper filtering rules:
    - 文档长度限制
    - 平均单词长度
    - 标点符号比例
    - 重复 n-gram 比例
    - "the", "be", "to" 等常见词存在性
    """
    pass
```

#### 与大规模预训练的关系
- 理解为什么 "Garbage in, garbage out"
- 数据质量直接影响模型能力

---

### Assignment 5: Alignment（对齐）
**主题**: 让模型与人类意图对齐

#### 核心实现内容

| 模块 | 描述 | 技术细节 |
|------|------|----------|
| **SFT (Supervised Fine-Tuning)** | 监督微调 | Instruction Following |
| **GRPO (Group Relative Policy Optimization)** | 强化学习对齐 | DeepSeek-R1 的核心算法 |
| **DPO (Direct Preference Optimization)** | 直接偏好优化 | 无需 RM 的对齐方法 |
| **Reward Modeling** | 奖励模型训练 | 偏好数据 → 奖励信号 |
| **MMLU/GSM8K 评估** | 模型能力评估 | 标准化测试 |

#### GRPO 核心公式

**优势函数计算**:
$$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$$

**GRPO-Clip 损失**:
$$L = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right)\right]$$

#### 可选内容（高级）
- **Safety Alignment**: 安全对齐
- **Instruction Tuning**: 指令微调
- **RLHF**: 基于人类反馈的强化学习

---

## 🔗 CS336 与 vLLM 的技术对应

| CS336 内容 | vLLM 对应实现 | 文件/目录 |
|------------|--------------|-----------|
| Transformer 架构 | 模型加载器 | `vllm/model_executor/models/` |
| FlashAttention | Attention Backend | `vllm/attention/` |
| RoPE 位置编码 | 模型层 | `vllm/model_executor/layers/rotary_embedding.py` |
| Tokenizer | Tokenizer 接口 | `vllm/transformers_utils/tokenizer.py` |
| 分布式训练 | Tensor/Pipeline Parallel | `vllm/distributed/` |
| 内存优化 | PagedAttention | `vllm/attention/backends/` |
| Continuous Batching | Scheduler | `vllm/core/scheduler.py` |

---

## 📖 Lectures 与 Assignments 对应

CS336 的 Lectures 提供了理论背景：

| Lecture | 主题 | 对应 Assignment |
|---------|------|----------------|
| Lecture 01-02 | Transformer 基础 | Assignment 1 |
| Lecture 06 | Systems, FlashAttention | Assignment 2 |
| Lecture 08 | 分布式训练 | Assignment 2 |
| Lecture 10 | 推理优化 (KV Cache, Batching) | 理解 vLLM |
| Lecture 12 | 数据处理 | Assignment 4 |
| Lecture 13-14 | Scaling Laws | Assignment 3 |
| Lecture 17 | Alignment | Assignment 5 |

---

## 🚀 建议的学习计划

### Phase 1: 基础 (2-3 周)
- [ ] 完成 Assignment 1 的核心模块
- [ ] 阅读 Lecture 01-02 的代码
- [ ] 在 TinyStories 上训练一个小模型

### Phase 2: 系统优化 (2-3 周)  
- [ ] 完成 Assignment 2 的 FlashAttention (PyTorch)
- [ ] 学习 Triton 基础，尝试 Triton 版本
- [ ] 阅读 vLLM 的 `docs/flash_attn_scratchpad.py` 理解核心思想
- [ ] 理解 DDP 和 Sharded Optimizer

### Phase 3: 深入 vLLM (2-3 周)
- [ ] 阅读 Lecture 10 关于推理优化
- [ ] 研究 vLLM 的 Scheduler 和 Block Manager
- [ ] 理解 PagedAttention 如何扩展 FlashAttention

### Phase 4: 高级主题 (2-3 周)
- [ ] 完成 Assignment 3 理解 Scaling Laws
- [ ] 完成 Assignment 4 了解数据工程
- [ ] 完成 Assignment 5 学习对齐技术

---

## 📁 目录结构说明

```
cs336/
├── assignment1-basics/          # 基础模块实现
│   ├── cs336_basics/            # 待实现的代码
│   ├── tests/                   # 测试用例
│   └── cs336_spring2025_assignment1_basics.pdf
│
├── assignment2-systems/         # 系统优化
│   ├── cs336-basics/            # Assignment 1 参考实现
│   ├── cs336_systems/           # 待实现的代码
│   └── cs336_spring2025_assignment2_systems.pdf
│
├── assignment3-scaling/         # Scaling Laws
│   ├── cs336_scaling/           # 分析代码
│   └── cs336_spring2025_assignment3_scaling.pdf
│
├── assignment4-data/            # 数据处理
│   ├── cs336_data/              # 待实现的代码
│   └── cs336_spring2025_assignment4_data.pdf
│
├── assignment5-alignment/       # 对齐
│   ├── cs336_alignment/         # 待实现的代码
│   ├── cs336_spring2025_assignment5_alignment.pdf
│   └── cs336_spring2025_assignment5_supplement_safety_rlhf.pdf  # 可选
│
└── spring2025-lectures/         # 课程代码示例
    ├── lecture_01.py            # Tokenization, Basics
    ├── lecture_02.py            # Transformer
    ├── lecture_06.py            # Systems, FlashAttention
    ├── lecture_08.py            # Distributed Training
    ├── lecture_10.py            # Inference Optimization
    ├── lecture_12.py            # Data
    ├── lecture_13.py            # Scaling Laws
    ├── lecture_14.py            # Scaling Laws (续)
    └── lecture_17.py            # Alignment
```

---

## 💡 学习技巧

1. **先跑通测试**: 每个 assignment 都有 pytest 测试，通过测试验证实现
   ```bash
   cd cs336/assignment1-basics
   uv run pytest tests/test_model.py -v
   ```

2. **对照 Lectures 代码**: `spring2025-lectures/` 目录下有详细的示例代码

3. **PDF 是关键**: 每个 assignment 的 PDF 文件包含详细的算法说明和数学公式

4. **结合 vLLM 源码**: 完成 Assignment 2 后，阅读 vLLM 的 attention 实现会更有收获

5. **使用 WSL/Linux**: 某些 Triton 相关功能需要在 Linux 环境运行

---

## 📚 参考资源

- [Stanford CS336 Course Page](https://stanford-cs336.github.io/spring2025/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [DeepSeek-R1 (GRPO)](https://arxiv.org/abs/2501.12948)
