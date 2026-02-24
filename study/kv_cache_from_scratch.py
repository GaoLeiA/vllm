"""
KV Cache 从零实现 — 100% 可运行的教学代码
=============================================

目标：彻底搞懂 KV Cache 在推理中到底干了什么。

运行方式：
    python study/kv_cache_from_scratch.py

你会看到：
    1. 不用 KV Cache 的推理（每次重算所有 token 的 K、V）
    2. 用 KV Cache 的推理（只算新 token 的 K、V，历史的从 Cache 里取）
    3. 两者结果完全一致 ✅
    4. KV Cache 版本快得多 ⚡
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

torch.manual_seed(42)

# ================================================================
# 第 0 部分：超参数（故意设得很小，方便理解）
# ================================================================
VOCAB_SIZE = 128       # 词表大小
D_MODEL = 64           # 模型维度
NUM_HEADS = 4          # 注意力头数
D_HEAD = D_MODEL // NUM_HEADS  # 每个头的维度 = 16
NUM_LAYERS = 2         # Transformer 层数
MAX_SEQ_LEN = 32       # 最大序列长度


# ================================================================
# 第 1 部分：最简单的 Causal Self-Attention（不用 KV Cache）
# ================================================================
class CausalSelfAttention(nn.Module):
    """标准因果自注意力，每次都重新计算所有 token 的 Q、K、V"""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        B, N, D = x.shape

        # ============================================
        # 对 **所有** token 计算 Q, K, V
        # 这就是没有 KV Cache 的代价：
        #   每生成一个新 token，前面的 K、V 都要重算！
        # ============================================
        Q = self.q_proj(x).view(B, N, NUM_HEADS, D_HEAD).transpose(1, 2)  # (B, H, N, D_head)
        K = self.k_proj(x).view(B, N, NUM_HEADS, D_HEAD).transpose(1, 2)  # (B, H, N, D_head)
        V = self.v_proj(x).view(B, N, NUM_HEADS, D_HEAD).transpose(1, 2)  # (B, H, N, D_head)

        # 标准 Scaled Dot-Product Attention + Causal Mask
        scale = 1.0 / math.sqrt(D_HEAD)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # 因果掩码：每个 token 只能看到自己和之前的 token
        causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N, D_head)

        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        return self.o_proj(out)


# ================================================================
# 第 2 部分：带 KV Cache 的 Causal Self-Attention（核心！）
# ================================================================
class CausalSelfAttentionWithKVCache(nn.Module):
    """
    带 KV Cache 的因果自注意力。

    核心思想：
        K 和 V 只和输入 token 有关，和其他 token 无关。
        所以历史 token 的 K、V 算过一次后，存起来就行了，不需要重算！

        - Prefill 阶段：处理整个 prompt，计算所有 token 的 K、V，存入 Cache
        - Decode 阶段：每次只处理 1 个新 token，
                       Q 只有 1 行，K 和 V 从 Cache 取出 + 追加新的
    """

    def __init__(self):
        super().__init__()
        # 权重和不带 Cache 的版本 **完全一样**
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x, kv_cache=None):
        """
        x: (batch, seq_len, d_model)
           - Prefill 时 seq_len = prompt 长度
           - Decode 时 seq_len = 1（只有新 token）

        kv_cache: None 或 (cached_K, cached_V)
           - cached_K: (batch, heads, past_len, d_head)
           - cached_V: (batch, heads, past_len, d_head)

        返回: (output, new_kv_cache)
        """
        B, N, D = x.shape

        # ============================================
        # 只对 **新 token** 计算 Q, K, V
        # ============================================
        Q = self.q_proj(x).view(B, N, NUM_HEADS, D_HEAD).transpose(1, 2)  # (B, H, N_new, D_head)
        K_new = self.k_proj(x).view(B, N, NUM_HEADS, D_HEAD).transpose(1, 2)  # (B, H, N_new, D_head)
        V_new = self.v_proj(x).view(B, N, NUM_HEADS, D_HEAD).transpose(1, 2)  # (B, H, N_new, D_head)

        # ============================================
        # 关键操作：把新的 K、V 追加到 Cache 后面
        # ============================================
        if kv_cache is not None:
            K_cached, V_cached = kv_cache
            # 拼接：[历史的 K | 新的 K]
            K = torch.cat([K_cached, K_new], dim=2)  # (B, H, past+new, D_head)
            V = torch.cat([V_cached, V_new], dim=2)  # (B, H, past+new, D_head)
        else:
            # 第一次（Prefill），没有历史
            K = K_new
            V = V_new

        # ============================================
        # 保存更新后的 Cache（给下一步用）
        # ============================================
        new_kv_cache = (K, V)

        # ============================================
        # Attention 计算
        # Q 的 seq_len 可能是 1（Decode）或 N（Prefill）
        # K、V 的 seq_len 是 past_len + N_new
        #
        # 这就是为什么 Decode 时 Attention 很快：
        #   Q: (B, H, 1, D_head)      ← 只有 1 行！
        #   K: (B, H, past+1, D_head)  ← 但 K 有很多行（从 Cache 来的）
        #   scores: (B, H, 1, past+1)  ← 只算 1 行的分数！
        # ============================================
        total_len = K.shape[2]
        scale = 1.0 / math.sqrt(D_HEAD)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N_new, total_len)

        # 因果掩码：新 token 的位置是 [total_len - N_new, total_len)
        # 它们可以看到所有 <= 自己位置的 token
        q_positions = torch.arange(total_len - N, total_len, device=x.device)[:, None]  # (N_new, 1)
        k_positions = torch.arange(total_len, device=x.device)[None, :]                  # (1, total_len)
        causal_mask = q_positions >= k_positions  # (N_new, total_len)
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N_new, D_head)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.o_proj(out), new_kv_cache


# ================================================================
# 第 3 部分：极简 Transformer（用于测试）
# ================================================================
class MiniTransformer(nn.Module):
    """2 层 Transformer + Embedding + LM Head"""

    def __init__(self, use_kv_cache=False):
        super().__init__()
        self.use_kv_cache = use_kv_cache
        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL)

        if use_kv_cache:
            self.layers = nn.ModuleList([CausalSelfAttentionWithKVCache() for _ in range(NUM_LAYERS)])
        else:
            self.layers = nn.ModuleList([CausalSelfAttention() for _ in range(NUM_LAYERS)])

        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, token_ids, past_kv_caches=None):
        """
        token_ids: (batch, seq_len) — 输入 token ID
        past_kv_caches: list of kv_cache per layer (KV Cache 模式才用)

        返回:
            logits: (batch, seq_len, vocab_size)
            new_kv_caches: 更新后的 KV Cache（如果有的话）
        """
        x = self.embed(token_ids)  # (B, N, D)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.use_kv_cache:
                past_kv = past_kv_caches[i] if past_kv_caches else None
                x, new_kv = layer(x, kv_cache=past_kv)
                new_kv_caches.append(new_kv)
            else:
                x = layer(x)

        logits = self.lm_head(x)  # (B, N, vocab_size)
        return logits, new_kv_caches if self.use_kv_cache else None


# ================================================================
# 第 4 部分：共享权重（确保两个模型完全相同）
# ================================================================
def copy_weights(src: MiniTransformer, dst: MiniTransformer):
    """把 src 的权重复制到 dst，确保两个模型输出一致"""
    dst.embed.load_state_dict(src.embed.state_dict())
    dst.lm_head.load_state_dict(src.lm_head.state_dict())
    for src_layer, dst_layer in zip(src.layers, dst.layers):
        dst_layer.load_state_dict(src_layer.state_dict())


# ================================================================
# 第 5 部分：推理对比实验
# ================================================================
def generate_without_kv_cache(model, prompt_ids, num_new_tokens):
    """
    不用 KV Cache 的生成：
    每生成一个新 token，都要把 **整个序列** 重新过一遍模型。

    prompt = [A, B, C]
    第 1 步: model([A, B, C])           → 取最后一个 token 的 logits → D
    第 2 步: model([A, B, C, D])        → 取最后一个 token 的 logits → E
    第 3 步: model([A, B, C, D, E])     → 取最后一个 token 的 logits → F
                   ↑ 每次都重算全部！浪费！
    """
    generated = prompt_ids.clone()

    for step in range(num_new_tokens):
        logits, _ = model(generated)
        next_token_logits = logits[:, -1, :]  # 只看最后一个位置
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def generate_with_kv_cache(model, prompt_ids, num_new_tokens):
    """
    用 KV Cache 的生成：
    Prefill: 只跑一次 prompt，把 K、V 存入 Cache。
    Decode:  每步只输入 1 个新 token，K、V 从 Cache 取。

    prompt = [A, B, C]
    Prefill: model([A, B, C])  → Cache 存了 K_A, K_B, K_C (和 V)
    Decode 1: model([D], cache) → Cache 追加 K_D → 输出 E
    Decode 2: model([E], cache) → Cache 追加 K_E → 输出 F
                    ↑ 每次只算 1 个 token！超快！
    """
    generated = prompt_ids.clone()

    # ==========================================
    # Prefill 阶段：处理整个 prompt
    # ==========================================
    logits, kv_caches = model(prompt_ids)
    next_token_logits = logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)

    # ==========================================
    # Decode 阶段：每次只输入 1 个新 token
    # ==========================================
    for step in range(num_new_tokens - 1):
        # 注意：只送入最后一个 token！
        logits, kv_caches = model(next_token, past_kv_caches=kv_caches)
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


# ================================================================
# 第 6 部分：运行实验！
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("KV Cache 从零实现 — 教学演示")
    print("=" * 60)

    # 创建两个模型：一个不带 Cache，一个带 Cache
    model_no_cache = MiniTransformer(use_kv_cache=False)
    model_with_cache = MiniTransformer(use_kv_cache=True)

    # 复制权重，确保两个模型完全相同
    copy_weights(model_no_cache, model_with_cache)

    # 构造一个 prompt
    prompt = torch.randint(0, VOCAB_SIZE, (1, 8))  # batch=1, prompt_len=8
    num_new_tokens = 10
    print(f"\nPrompt (长度={prompt.shape[1]}): {prompt[0].tolist()}")
    print(f"要生成的新 token 数: {num_new_tokens}")

    # ==========================================
    # 实验 1：不用 KV Cache
    # ==========================================
    print("\n" + "-" * 40)
    print("🐌 不用 KV Cache 的推理:")
    print("-" * 40)

    with torch.no_grad():
        t0 = time.perf_counter()
        result_no_cache = generate_without_kv_cache(model_no_cache, prompt, num_new_tokens)
        t1 = time.perf_counter()

    print(f"  生成结果: {result_no_cache[0].tolist()}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    print(f"  每步都重算整个序列的 K、V（浪费！）")

    # ==========================================
    # 实验 2：用 KV Cache
    # ==========================================
    print("\n" + "-" * 40)
    print("⚡ 用 KV Cache 的推理:")
    print("-" * 40)

    with torch.no_grad():
        t0 = time.perf_counter()
        result_with_cache = generate_with_kv_cache(model_with_cache, prompt, num_new_tokens)
        t1 = time.perf_counter()

    print(f"  生成结果: {result_with_cache[0].tolist()}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    print(f"  Decode 阶段每步只算 1 个 token 的 K、V！")

    # ==========================================
    # 验证：两者结果必须完全一致！
    # ==========================================
    print("\n" + "-" * 40)
    print("🔍 正确性验证:")
    print("-" * 40)

    match = torch.equal(result_no_cache, result_with_cache)
    if match:
        print("  ✅ 两种方式的生成结果 **完全一致**！")
        print("  证明 KV Cache 只是优化了速度，不影响结果。")
    else:
        print("  ❌ 结果不一致，代码有 bug！")
        print(f"  无 Cache: {result_no_cache[0].tolist()}")
        print(f"  有 Cache: {result_with_cache[0].tolist()}")

    # ==========================================
    # 可视化 KV Cache 的大小变化
    # ==========================================
    print("\n" + "-" * 40)
    print("📊 KV Cache 大小变化 (Decode 阶段):")
    print("-" * 40)

    with torch.no_grad():
        _, kv_caches = model_with_cache(prompt)
        next_token = torch.randint(0, VOCAB_SIZE, (1, 1))

        for step in range(5):
            _, kv_caches = model_with_cache(next_token, past_kv_caches=kv_caches)
            cache_k, cache_v = kv_caches[0]  # 第一层的 Cache
            print(f"  Step {step+1}: "
                  f"Cache K shape = {list(cache_k.shape)}, "
                  f"Cache V shape = {list(cache_v.shape)}, "
                  f"存了 {cache_k.shape[2]} 个 token 的 K/V")
            next_token = torch.randint(0, VOCAB_SIZE, (1, 1))

    print()
    print("=" * 60)
    print("总结:")
    print("  • KV Cache 就是一个 list，存每层的 (K, V) 历史")
    print("  • Prefill: 一次算完 prompt 的 K、V，存入 Cache")
    print("  • Decode: 每步只算 1 个新 token 的 Q、K、V")
    print("    Q 只有 1 行 → Attention 计算量从 O(N²) 降到 O(N)")
    print("  • vLLM 的 PagedAttention 就是在 KV Cache 基础上，")
    print("    把连续存储改成分页存储（像操作系统的虚拟内存）")
    print("=" * 60)
