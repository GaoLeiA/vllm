# CS336 Assignment 5: Alignment - 学习计划

> **目标**: 理解 LLM 对齐技术 —— 如何让模型按照人类意图生成有用且安全的回复

---

## 📋 Assignment 5 任务概览

| 模块 | 任务描述 | 难度 | 代码接口 |
|------|---------|------|---------|
| **Tokenize Prompt/Output** | 分词 + response mask 构建 | ⭐⭐ | `run_tokenize_prompt_and_output()` |
| **Masked Mean** | 带 mask 的均值计算 | ⭐ | `run_masked_mean()` |
| **Masked Normalize** | 带 mask 的归一化 | ⭐ | `run_masked_normalize()` |
| **Response Log-Probs** | 计算 response 的对数概率 | ⭐⭐ | `run_get_response_log_probs()` |
| **Entropy** | 计算 logits 的熵 | ⭐ | `run_compute_entropy()` |
| **SFT Loss** | 监督微调损失 + 梯度累积 | ⭐⭐ | `run_sft_microbatch_train_step()` |
| **Naive Policy Gradient** | 基础策略梯度损失 | ⭐⭐ | `run_compute_naive_policy_gradient_loss()` |
| **GRPO Group Rewards** | GRPO 组归一化奖励 | ⭐⭐⭐ | `run_compute_group_normalized_rewards()` |
| **GRPO-Clip Loss** | PPO 风格的裁剪损失 | ⭐⭐⭐ | `run_compute_grpo_clip_loss()` |
| **GRPO Train Step** | GRPO 完整训练步 | ⭐⭐⭐ | `run_grpo_microbatch_train_step()` |
| **DPO Loss** | Direct Preference Optimization | ⭐⭐⭐ | `run_compute_per_instance_dpo_loss()` |
| **Packed SFT Dataset** | 打包式 SFT 数据集 | ⭐⭐ | `get_packed_sft_dataset()` |
| **MMLU/GSM8K 解析** | 评估结果解析 | ⭐⭐ | `run_parse_mmlu/gsm8k_response()` |

---

## 🎯 核心知识点

### 1. 对齐 (Alignment) 全景

```
预训练模型 (Base LLM)
    │
    ├── SFT (Supervised Fine-Tuning): 模仿高质量回复
    │     Input: (prompt, good_response) pairs
    │     Loss: -log P(response | prompt) — 标准 cross-entropy
    │
    ├── RLHF (RL from Human Feedback): 强化学习优化
    │     │
    │     ├── PPO: Proximal Policy Optimization
    │     │     需要: reward model + reference policy
    │     │     Loss: clip(π/π_old, 1±ε) × advantage
    │     │
    │     └── GRPO: Group Relative Policy Optimization  ⭐ (DeepSeek)
    │           不需要 reward model!
    │           为每个 prompt 生成 G 个回复
    │           用组内相对奖励作为 advantage
    │
    └── DPO (Direct Preference Optimization): 直接偏好优化
          Input: (prompt, chosen, rejected) triplets
          Loss: -log σ(β × (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))
```

---

### 2. SFT (Supervised Fine-Tuning)

```python
# SFT 的核心: 只对 response 部分计算损失
# 
# Input:  [<prompt tokens> | <response tokens>]
# Mask:   [0  0  0  0  0   | 1  1  1  1  1   ]
#          ↑ 不计算损失       ↑ 计算损失

# 标准 SFT 损失:
def sft_loss(logits, labels, response_mask):
    # 计算每个 token 的 cross-entropy
    token_loss = F.cross_entropy(logits, labels, reduction='none')
    # 只对 response token 取平均
    loss = (token_loss * response_mask).sum() / response_mask.sum()
    return loss
```

**Tokenize + Mask 构建**:
```python
# 输入: prompt_strs = ["What is AI?"], output_strs = ["AI is..."]
# 输出:
#   input_ids:     [prompt_token_ids + response_token_ids][:-1]  (去掉最后一个)
#   labels:        [prompt_token_ids + response_token_ids][1:]   (去掉第一个，shifted)
#   response_mask: [0, 0, ..., 0, 1, 1, ..., 1]  (response 位置为 1)
```

**梯度累积 (Gradient Accumulation)**:
```python
# 当 GPU 显存不够时，将大 batch 拆成小 microbatch
# 每个 microbatch 计算 loss 并调用 backward，但不 step
# 累积 N 次后才 optimizer.step()

for microbatch in split(batch, gradient_accumulation_steps):
    loss = compute_loss(microbatch) / gradient_accumulation_steps
    loss.backward()  # 梯度累积
optimizer.step()     # 一次更新
optimizer.zero_grad()
```

---

### 3. GRPO (Group Relative Policy Optimization) ⭐

**DeepSeek-R1 的核心训练算法**:

```
GRPO 流程:
┌─────────────────────────────────────────────────────────┐
│ 对每个 prompt:                                          │
│   1. 用当前策略生成 G 个回复 (group_size)                │
│   2. 用 reward_fn 给每个回复打分                        │
│   3. 组内归一化: advantage = (r - mean(r)) / std(r)     │
│   4. 计算策略梯度损失 (支持三种变体):                    │
│      a. no_baseline:           -r × log π(response)     │
│      b. reinforce_with_baseline: -advantage × log π     │
│      c. grpo_clip:             clip ratio × advantage   │
└─────────────────────────────────────────────────────────┘
```

```python
# 组归一化奖励
def compute_group_normalized_rewards(rewards, group_size, eps=1e-8):
    # rewards: [g1_r1, g1_r2, ..., g1_rG, g2_r1, g2_r2, ..., g2_rG, ...]
    rewards = rewards.reshape(-1, group_size)  # (num_prompts, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean) / (std + eps)
    return advantages.flatten()

# GRPO-Clip Loss (类似 PPO)
def grpo_clip_loss(advantages, log_probs, old_log_probs, cliprange):
    ratio = torch.exp(log_probs - old_log_probs)  # π/π_old
    clipped_ratio = torch.clamp(ratio, 1-cliprange, 1+cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss
```

**Dr. GRPO 变体** (用于本 assignment):
- 不用 token 级平均，而是 sequence 级求和后除以常数
- `run_masked_normalize` 实现了这个归一化

---

### 4. DPO (Direct Preference Optimization)

```python
# DPO 直接用偏好数据优化策略，不需要训练 reward model

# DPO 损失:
def dpo_loss(policy, ref_model, prompt, chosen, rejected, beta):
    # 计算 policy 和 reference model 的 log-probs
    pi_chosen = log_prob(policy, prompt + chosen)
    pi_rejected = log_prob(policy, prompt + rejected)
    ref_chosen = log_prob(ref_model, prompt + chosen)
    ref_rejected = log_prob(ref_model, prompt + rejected)
    
    # DPO 目标: 让 chosen 的 "优势" 大于 rejected
    log_ratio_chosen = pi_chosen - ref_chosen
    log_ratio_rejected = pi_rejected - ref_rejected
    
    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss
```

---

### 5. 评估 (MMLU, GSM8K)

```python
# MMLU: 多选题评估
# 解析模型输出中的选项字母 (A/B/C/D)
def parse_mmlu_response(example, model_output):
    # 找到 model_output 中的选项字母
    # 可以用正则匹配: "The answer is (A|B|C|D)"
    match = re.search(r'[ABCD]', model_output)
    return match.group() if match else None

# GSM8K: 数学题评估
# 提取最后出现的数字作为答案
def parse_gsm8k_response(model_output):
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', model_output)
    return numbers[-1] if numbers else None
```

---

## 🎯 学习路径

### Phase 1: SFT 基础 (3小时)

**任务**: 实现 tokenization、masked loss、梯度累积

1. `run_tokenize_prompt_and_output()` — prompt/response 分词 + mask
2. `run_masked_mean()` — 带 mask 的均值
3. `run_masked_normalize()` — Dr. GRPO 归一化
4. `run_get_response_log_probs()` — 计算 log-probs
5. `run_compute_entropy()` — logits 的熵
6. `run_sft_microbatch_train_step()` — SFT 训练步

**💡 关键概念**:
- 为什么只在 response 上计算损失？→ prompt 是给定的，不需要生成
- 为什么需要梯度累积？→ 大 batch 尺寸 + 有限 GPU 显存

---

### Phase 2: GRPO 策略梯度 (4小时)

**任务**: 实现 GRPO 的奖励计算和策略梯度

1. `run_compute_group_normalized_rewards()` — 组归一化
2. `run_compute_naive_policy_gradient_loss()` — 基础策略梯度
3. `run_compute_grpo_clip_loss()` — PPO-style 裁剪
4. `run_compute_policy_gradient_loss()` — 统一接口
5. `run_grpo_microbatch_train_step()` — GRPO 训练步

**💡 关键概念**:
- **为什么需要 baseline？** 减小方差，加速收敛
- **为什么需要 clip？** 防止策略更新过大 (Trustregion)
- **GRPO vs PPO**: GRPO 不需要 critic/reward model

---

### Phase 3: DPO + 评估 (3小时)

**任务**: Direct Preference Optimization + 模型评估

1. `run_compute_per_instance_dpo_loss()` — DPO 损失
2. `get_packed_sft_dataset()` — 打包数据集
3. `run_iterate_batches()` — batch 迭代器
4. `run_parse_mmlu_response()` — MMLU 解析
5. `run_parse_gsm8k_response()` — GSM8K 解析

**💡 关键概念**:
- **DPO vs RLHF**: DPO 更简单 (不需要 reward model)，但表达能力可能较弱
- **Packed Dataset**: 将多个短序列打包到固定长度以提高效率

---

## 🔗 与 vLLM 的连接 ⭐ (高度相关)

Assignment 5 与 vLLM **高度相关**:

| 对齐概念 | vLLM 直接关联 |
|---------|-------------|
| **SFT 推理** | vLLM 服务的 SFT 模型 — 理解模型为什么能按指令回答 |
| **GRPO Rollout 生成** | 🔥 需要高效推理服务！vLLM 正是为此设计 |
| **DPO** | vLLM 服务对齐后的模型 |
| **MMLU/GSM8K 评估** | vLLM 作为推理后端加速评估 |
| **梯度累积** | 理解训练 vs 推理的内存权衡 |

**GRPO 与 vLLM 的深层联系**:
```
GRPO 训练循环:
┌──────────────────────────────────────────────────┐
│ 1. Rollout Phase (用 vLLM 加速!)                 │
│    - 用当前策略为每个 prompt 生成 G 个回复         │
│    - 这正是 vLLM 的 batch inference! 🚀           │
│                                                  │
│ 2. Scoring Phase                                 │
│    - 用 reward function 给回复打分                │
│                                                  │
│ 3. Training Phase                                │
│    - 计算策略梯度损失并更新模型                    │
│    - 需要用 reference policy 的 log-probs          │
│    - vLLM 也可以加速 reference model 推理! 🚀      │
└──────────────────────────────────────────────────┘

实际系统 (如 OpenRLHF, verl): 
  - 用 vLLM 做 rollout generation
  - 用 DeepSpeed/FSDP 做 training
  - 两者交替执行
```

---

## 📚 配套讲座

1. **Lecture 15** — SFT 基础: instruction tuning, data formatting
2. **Lecture 16** — RLHF: reward models, PPO
3. **Lecture 17** — DPO, GRPO, modern alignment methods
4. **Lecture 18** — Safety and evaluation

---

## 📚 必读论文

| 论文 | 重要性 | 核心贡献 |
|------|-------|---------|
| [InstructGPT (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) | ⭐⭐⭐ | RLHF 经典流程 |
| [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) | ⭐⭐⭐ | 无需 reward model 的对齐 |
| [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) | ⭐⭐⭐ | GRPO 算法 |
| [DeepSeekMath (2024)](https://arxiv.org/abs/2402.03300) | ⭐⭐⭐ | GRPO 原始论文 |
| [Dr. GRPO (2025)](https://arxiv.org/abs/2503.20783) | ⭐⭐ | GRPO 改进 |

---

## 🧪 测试命令

```bash
cd D:\projects\vllm\cs336\assignment5-alignment

# 安装依赖
uv sync

# 运行所有测试
uv run pytest tests/ -v

# 单独测试各模块
uv run pytest tests/test_data.py -v     # 数据处理
uv run pytest tests/test_sft.py -v      # SFT
uv run pytest tests/test_grpo.py -v     # GRPO
uv run pytest tests/test_dpo.py -v      # DPO
uv run pytest tests/test_metrics.py -v  # 评估
```

---

## 📅 学习时间安排

| 阶段 | 预计时间 | 产出 |
|------|---------|------|
| Phase 1: SFT 基础 | 3小时 | tokenization, SFT loss, gradient accumulation |
| Phase 2: GRPO 策略梯度 | 4小时 | GRPO rewards, clip loss, train step |
| Phase 3: DPO + 评估 | 3小时 | DPO loss, MMLU/GSM8K parsing |
| **阅读 DeepSeek-R1 论文** | 2小时 | 理解 GRPO 在 R1 中的应用 |
| **对照 vLLM GRPO 整合** | 2小时 | 理解推理服务在 RL 训练中的角色 |

**总计**: ~14小时

---

## ✅ 关键学习成果

完成 Assignment 5 后，您将理解：
- 🎓 **SFT → RLHF → DPO 的演进**: 对齐技术的发展脉络
- 🧠 **GRPO 算法详解**: DeepSeek-R1 的核心训练方法
- 🔄 **策略梯度 + Clipping**: PPO/GRPO 的数学原理
- 🆚 **DPO vs RLHF**: 各自的优缺点
- 🚀 **vLLM 在 RL 训练中的角色**: 为什么高效推理对 alignment 训练至关重要
- 📊 **模型评估**: MMLU/GSM8K 的评估方法

---

## 💡 学习建议

Assignment 5 是 **最前沿** 的内容 (GRPO 来自 2024-2025 年的论文)，建议:

1. **先理解 SFT** — 这是基础，概念最简单
2. **再学 REINFORCE** — 理解基础策略梯度
3. **然后学 GRPO** — 在 REINFORCE 基础上增加组归一化和裁剪
4. **最后学 DPO** — 独立的对齐方法
5. **读 DeepSeek-R1 论文** — 看 GRPO 如何在实际中训练推理模型
