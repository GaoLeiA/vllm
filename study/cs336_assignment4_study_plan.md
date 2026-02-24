# CS336 Assignment 4: Data - 学习计划

> **目标**: 理解 LLM 训练数据的处理流程 —— 从原始网页到干净的训练语料

---

## 📋 Assignment 4 任务概览

| 模块 | 任务描述 | 难度 | 代码接口 |
|------|---------|------|---------|
| **HTML 文本提取** | 从 HTML 字节中提取干净文本 | ⭐⭐ | `run_extract_text_from_html_bytes()` |
| **语言识别** | 识别文本语言 | ⭐ | `run_identify_language()` |
| **PII 脱敏** | 屏蔽邮箱、电话号码、IP 地址 | ⭐⭐ | `run_mask_emails/phones/ips()` |
| **NSFW 分类** | 检测不安全内容 | ⭐⭐ | `run_classify_nsfw()` |
| **有害言论分类** | 检测有害/仇恨言论 | ⭐⭐ | `run_classify_toxic_speech()` |
| **质量分类** | 评估文本质量 | ⭐⭐ | `run_classify_quality()` |
| **Gopher 质量过滤** | 基于规则的质量过滤 | ⭐⭐⭐ | `run_gopher_quality_filter()` |
| **精确行去重** | 按行精确去重 | ⭐⭐ | `run_exact_line_deduplication()` |
| **MinHash 去重** | 基于 LSH 的近似去重 | ⭐⭐⭐⭐ | `run_minhash_deduplication()` |

---

## 🎯 核心知识点

### 1. 数据处理 Pipeline 全景

```
                    LLM 数据处理 Pipeline
                    
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ 原始网页  │ →  │ 文本提取  │ →  │ 语言过滤  │ →  │ 质量过滤  │
  │ (HTML)   │    │ (Extract)│    │ (LangID) │    │ (Quality)│
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                        │
  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
  │ 训练数据  │ ←  │  去重    │ ←  │ PII脱敏  │ ←  │ 安全过滤 │
  │ (Clean)  │    │ (Dedup) │    │ (Mask)   │    │ (Safety)│
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

### 2. HTML 文本提取

```python
# 从 HTML 中提取干净文本
# 需要处理: script/style 标签移除, 空白规范化, boilerplate 移除
# 常用库: trafilatura, jusText, resiliparse

from trafilatura import extract
text = extract(html_bytes.decode('utf-8'))
```

---

### 3. PII (Personally Identifiable Information) 脱敏

```python
# 需要用正则表达式识别和替换:
# 1. 邮箱: user@example.com → |||EMAIL_ADDRESS|||
# 2. 电话号码: (123) 456-7890 → |||PHONE_NUMBER|||  
# 3. IP 地址: 192.168.1.1 → |||IP_ADDRESS|||

import re

# 邮箱匹配
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# 电话号码 (多种格式)
PHONE_PATTERNS = [
    r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
    r'\d{3}[-.]?\d{3}[-.]?\d{4}',
    # 更多格式...
]

# IP 地址
IP_PATTERN = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
```

---

### 4. 质量过滤

**Gopher Quality Filter** (DeepMind 的过滤标准):

```
Gopher 规则:
┌────────────────────────────────────────────────────────┐
│ 1. 文档必须包含 50-100,000 个词                         │
│ 2. 词的平均长度在 3-10 个字符之间                       │
│ 3. 90% 以上的行不能以省略号结尾                         │
│ 4. 至少 80% 的词必须包含至少一个字母字符                 │
│ 5. "the" 出现次数必须 >= 总词数的某个比例                │
│ 6. 省略号行比例 < 30%                                   │
│ 7. 字母数字比例 > 80%                                   │
│ 8. 不能有超多重复行/段落 (去重指标)                      │
└────────────────────────────────────────────────────────┘
```

---

### 5. 去重算法 ⭐ (最重要、最难的部分)

#### 5a. 精确行去重

```python
# 简单但有效: 对每行计算 hash，去除重复行
# 空间效率: 只存 hash 而非原始文本

import hashlib

seen_hashes = set()
for line in document:
    h = hashlib.sha256(line.encode()).hexdigest()
    if h not in seen_hashes:
        seen_hashes.add(h)
        output.write(line)
```

#### 5b. MinHash LSH 去重 ⭐⭐⭐

**核心思想**: 用 Locality-Sensitive Hashing 找到近似相似的文档

```
MinHash 算法流程:
┌─────────────────────────────────────────────────────────────┐
│ 1. Shingling: 将文档分成 n-gram 集合                        │
│    "hello world" → {"hel","ell","llo","lo ","o w","wor",...}│
│                                                             │
│ 2. MinHash: 用 k 个 hash 函数生成签名                       │
│    signature = [min(h₁(shingles)), min(h₂(shingles)), ...]  │
│    长度 k 的签名 ≈ Jaccard 相似度的近似                      │
│                                                             │
│ 3. LSH Banding: 将签名分成 b 个 band，每个 band r 个行      │
│    如果两个文档在任一 band 完全匹配 → 候选对                  │
│    P(match) ≈ 1 - (1 - s^r)^b                              │
│    其中 s = Jaccard 相似度                                   │
│                                                             │
│ 4. 候选验证: 对候选对计算真实 Jaccard 相似度                  │
│    如果 > threshold → 标记为重复                             │
└─────────────────────────────────────────────────────────────┘
```

**LSH 参数调优**:
```python
# num_hashes = num_bands * rows_per_band
# 例如: 128 hashes = 16 bands × 8 rows
# 
# threshold ≈ (1/num_bands)^(1/rows_per_band)
# 调大 bands → 更高召回 (更多候选对)
# 调大 rows  → 更高精度 (更少假阳性)
```

---

## 🎯 学习路径

### Phase 1: HTML 文本提取 + 语言识别 (1.5小时)

```python
# 推荐库:
# - trafilatura: 高质量 HTML 文本提取
# - fasttext: 高速语言识别

# 安装
# uv add trafilatura fasttext-wheel

# 语言识别
import fasttext
model = fasttext.load_model('lid.176.bin')
predictions = model.predict(text)
# 返回: (('__label__en',), array([0.95]))
```

---

### Phase 2: PII 脱敏 (1.5小时)

正则表达式实战 —— 识别和替换敏感信息

**关键挑战**:
- 电话号码格式多样 (国际格式、带/不带括号...)
- 需要处理边界情况 (不要误匹配正常数字)
- 替换时需要记录替换次数

---

### Phase 3: 安全与质量分类 (2小时)

```python
# NSFW 和毒性检测通常使用预训练分类器
# 例如:
# - fasttext 分类器 (轻量级)
# - HuggingFace 上的 toxicity 检测模型

# 质量分类
# 可以用简单的启发式规则 (Gopher) 或训练分类器
```

---

### Phase 4: 精确去重 (1.5小时)

```python
# 输入: 多个文件，每个文件包含多个文档
# 输出: 去重后的文件 (去除重复行)
# 
# 挑战:
# - 跨文件去重 (全局 hash set)
# - 空间效率 (只存 hash)
# - 处理空行和空白
```

---

### Phase 5: MinHash 去重 ⭐ (4小时)

这是 Assignment 4 最核心最难的部分:

```python
# 实现步骤:
# 1. 文档 → n-gram 集合 (shingling)
# 2. n-gram 集合 → MinHash 签名
# 3. 签名 → LSH bands → 候选对
# 4. 候选对 → 验证 → 去重

def minhash_signature(shingles, num_hashes):
    """计算 MinHash 签名"""
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            h = hash_fn(shingle, seed=i)  # 不同 seed 的 hash
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def lsh_candidates(signatures, num_bands, rows_per_band):
    """LSH 找候选对"""
    buckets = defaultdict(set)
    for doc_id, sig in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            band = tuple(sig[start:start + rows_per_band])
            bucket_key = (band_idx, hash(band))
            buckets[bucket_key].add(doc_id)
    # 同一个 bucket 的文档是候选对
    ...
```

---

## 🔗 与 vLLM 的连接

虽然 vLLM 是推理框架，不直接处理数据，但理解数据处理有助于:

| 数据概念 | vLLM 间接影响 |
|---------|-------------|
| 数据质量 | 模型质量直接影响推理效果 |
| Tokenizer 兼容性 | vLLM 需要使用正确的 tokenizer |
| 数据规模 | 影响模型大小选择 → 影响 vLLM 资源需求 |
| 安全过滤 | vLLM 输出的安全性取决于训练数据 |

---

## 📚 配套讲座

1. **Lecture 12** — Data basics: web crawling, filtering
2. **Lecture 13** — Data deduplication, quality assessment
3. **Lecture 14** — Advanced data techniques

---

## 🧪 测试命令

```bash
cd D:\projects\vllm\cs336\assignment4-data

# 安装依赖
uv sync

# 运行所有测试
uv run pytest tests/ -v

# 单独测试各模块
uv run pytest tests/test_extract.py -v        # HTML 提取
uv run pytest tests/test_langid.py -v         # 语言识别
uv run pytest tests/test_pii.py -v            # PII 脱敏
uv run pytest tests/test_toxicity.py -v       # 有害内容
uv run pytest tests/test_quality.py -v        # 质量过滤
uv run pytest tests/test_deduplication.py -v  # 去重 (MinHash + Exact)
```

---

## 📅 学习时间安排

| 阶段 | 预计时间 | 产出 |
|------|---------|------|
| Phase 1: HTML 提取 + 语言识别 | 1.5小时 | text extraction, langid |
| Phase 2: PII 脱敏 | 1.5小时 | email/phone/ip masking |
| Phase 3: 安全与质量分类 | 2小时 | NSFW, toxic, quality classifiers |
| Phase 4: 精确去重 | 1.5小时 | exact line dedup |
| Phase 5: MinHash 去重 | 4小时 | MinHash LSH pipeline |

**总计**: ~10.5小时

---

## ✅ 关键学习成果

完成 Assignment 4 后，您将理解：
- 🌐 **数据 Pipeline**: 从原始网页到训练数据的完整流程
- 🔍 **去重技术**: LSH/MinHash 的原理和实现 (广泛应用于搜索引擎)
- 🛡️ **数据安全**: PII 脱敏、有害内容过滤
- 📊 **数据质量**: 如何评估和过滤低质量数据
- 🏭 **工程实践**: 大规模数据处理的效率考量
