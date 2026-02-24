### RMSNorm vs LayerNorm

------

## 先看公式

**LayerNorm**: $$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

需要计算：

1. 均值 $\mu = \frac{1}{d}\sum x_i$
2. 方差 $\sigma^2 = \frac{1}{d}\sum (x_i - \mu)^2$
3. 归一化
4. 缩放 $\gamma$ + 偏移 $\beta$（两组可学习参数）

**RMSNorm**: $$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$$

需要计算：

1. 均方根 $\text{RMS} = \sqrt{\frac{1}{d}\sum x_i^2}$
2. 归一化
3. 只有缩放 $\gamma$（一组可学习参数）

------

## 优势 1：更快（计算量少）

| 操作                           |      LayerNorm       |      RMSNorm      |
| :----------------------------- | :------------------: | :---------------: |
| 求均值 $\mu$                   |        ✅ 需要        |     ❌ 不需要      |
| 减去均值 $x - \mu$             |        ✅ 需要        |     ❌ 不需要      |
| 求方差 $\sigma^2$              |        ✅ 需要        |     ❌ 不需要      |
| 求平方和 $\sum x^2$            |          ❌           |      ✅ 需要       |
| 偏置参数 $\beta$               |         ✅ 有         |      ❌ 没有       |
| **Kernel 中的 Reduction 次数** | **2 次** (均值+方差) | **1 次** (平方和) |

在 GPU 上，**Reduction 操作**（所有线程协作求和）是最贵的部分。RMSNorm 少了一次 Reduction，实测快 **10-15%**。

------

## 优势 2：效果一样好（甚至更好）

[RMSNorm 原论文 (2019)](https://arxiv.org/abs/1910.07467) 的核心发现：

> LayerNorm 的成功主要归功于**缩放不变性**（Re-scaling Invariance），而不是**平移不变性**（Re-centering，即减去均值）。

实验证明：

- 去掉"减均值"这一步，模型效果几乎不变
- 去掉"除以标准差"这一步，模型效果明显下降

说明**归一化幅度**（让向量长度稳定）才是关键，**去中心化**（减均值）是多余的。

------

## 优势 3：参数更少

- LayerNorm: $\gamma$ + $\beta$（$2d$ 个参数）
- RMSNorm: 只有 $\gamma$（$d$ 个参数）

对于 LLaMA-70B（$d = 8192$, 80 层, 每层 2 个 Norm）：

- 节省参数：$80 \times 2 \times 8192 = 1.3M$ 个参数
- 虽然占比很小，但在推理时少一次加法也是优化

------

## 为什么现代 LLM 全用 RMSNorm？

| 模型                    | Norm 类型   |
| :---------------------- | :---------- |
| GPT-2 (2019)            | LayerNorm   |
| BERT (2018)             | LayerNorm   |
| LLaMA 1/2/3 (2023-2024) | **RMSNorm** |
| Qwen 1/2/3 (2023-2024)  | **RMSNorm** |
| Mistral/Mixtral         | **RMSNorm** |
| Gemma                   | **RMSNorm** |

**一句话总结**：RMSNorm 去掉了 LayerNorm 中不重要的"减均值"操作，保留了核心的"缩放归一化"，**更快、更简单、效果一样好**。在追求效率的 LLM 时代，没有理由用 LayerNorm。