#!/usr/bin/env python3
"""
vLLM Prefill 和 Decode 阶段完整流程演示

这个脚本模拟了 LLM 推理的两个核心阶段：
1. Prefill 阶段：处理完整的 prompt，生成 KV Cache
2. Decode 阶段：逐 token 生成，使用已缓存的 KV

以 Qwen3-4B 模型配置为例，展示 TP=2 的并行计算
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. 模型配置 (基于 Qwen3-4B)
# ============================================================================

@dataclass
class Qwen3Config:
    """Qwen3-4B 模型配置"""
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA: Grouped Query Attention
    head_dim: int = 128
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 40960
    
    # Tensor Parallelism 配置
    tp_size: int = 2  # 默认 TP=2
    
    @property
    def num_heads_per_tp(self) -> int:
        """每个 TP rank 的 attention heads 数量"""
        return self.num_attention_heads // self.tp_size
    
    @property
    def num_kv_heads_per_tp(self) -> int:
        """每个 TP rank 的 KV heads 数量"""
        return self.num_key_value_heads // self.tp_size


# ============================================================================
# 2. KV Cache 管理器 (Paged Attention 风格)
# ============================================================================

class PagedKVCache:
    """
    Paged KV Cache 管理器
    
    模拟 vLLM 的 Paged Attention:
    - KV Cache 被分成固定大小的 blocks
    - 每个请求动态分配 blocks
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # 预分配 KV Cache 存储
        # 形状: [num_blocks, num_kv_heads, block_size, head_dim]
        self.k_cache = torch.zeros(
            (num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (num_blocks, num_kv_heads, block_size, head_dim),
            dtype=dtype, device=device
        )
        
        # Block 分配表
        self.free_blocks = list(range(num_blocks))
        self.block_tables: dict[str, list[int]] = {}  # request_id -> block_ids
        
    def allocate_blocks(self, request_id: str, num_tokens: int) -> list[int]:
        """为请求分配所需的 blocks"""
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("KV Cache 空间不足!")
        
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)
        
        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
        self.block_tables[request_id].extend(allocated)
        
        return allocated
    
    def write_kv(
        self,
        request_id: str,
        key: torch.Tensor,     # [seq_len, num_kv_heads, head_dim]
        value: torch.Tensor,   # [seq_len, num_kv_heads, head_dim]
        start_pos: int,
    ):
        """将 K, V 写入缓存"""
        seq_len = key.shape[0]
        block_ids = self.block_tables[request_id]
        
        for i in range(seq_len):
            pos = start_pos + i
            block_idx = pos // self.block_size
            offset_in_block = pos % self.block_size
            
            block_id = block_ids[block_idx]
            self.k_cache[block_id, :, offset_in_block, :] = key[i]
            self.v_cache[block_id, :, offset_in_block, :] = value[i]
    
    def read_kv(
        self,
        request_id: str,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """读取完整的 KV Cache"""
        block_ids = self.block_tables[request_id]
        
        k_out = []
        v_out = []
        
        for pos in range(seq_len):
            block_idx = pos // self.block_size
            offset_in_block = pos % self.block_size
            block_id = block_ids[block_idx]
            
            k_out.append(self.k_cache[block_id, :, offset_in_block, :])
            v_out.append(self.v_cache[block_id, :, offset_in_block, :])
        
        return (
            torch.stack(k_out, dim=0),  # [seq_len, num_kv_heads, head_dim]
            torch.stack(v_out, dim=0),  # [seq_len, num_kv_heads, head_dim]
        )
    
    def free(self, request_id: str):
        """释放请求的 blocks"""
        if request_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[request_id])
            del self.block_tables[request_id]


# ============================================================================
# 3. RoPE (Rotary Position Embedding)
# ============================================================================

def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 1000000.0,
    device: str = "cuda",
) -> torch.Tensor:
    """预计算 RoPE 的频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,  # [seq_len, num_heads, head_dim]
    xk: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,  # [seq_len]
) -> tuple[torch.Tensor, torch.Tensor]:
    """应用 RoPE"""
    # 选择对应位置的频率
    freqs = freqs_cis[positions]  # [seq_len, head_dim/2]
    
    # 将 tensor 转换为复数形式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 应用旋转
    freqs = freqs.unsqueeze(1)  # [seq_len, 1, head_dim/2]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================================
# 4. Attention 层 (支持 GQA + RoPE + QK Norm)
# ============================================================================

class Qwen3Attention(nn.Module):
    """
    Qwen3 风格的 Attention 层
    
    特点:
    - Grouped Query Attention (GQA)
    - QK Norm (RMSNorm on Q and K)
    - Rotary Position Embedding (RoPE)
    - 支持 Tensor Parallelism
    """
    
    def __init__(self, config: Qwen3Config, layer_idx: int, tp_rank: int = 0):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.tp_rank = tp_rank
        self.tp_size = config.tp_size
        
        # TP 分片后的参数
        self.num_heads = config.num_heads_per_tp
        self.num_kv_heads = config.num_kv_heads_per_tp
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        self.scaling = self.head_dim ** -0.5
        
        # QKV Projection (Column Parallel)
        # 每个 TP rank 持有 Q, K, V 的一部分
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        
        # Output Projection (Row Parallel)
        self.o_proj = nn.Linear(
            self.q_size,  # 本地 input
            self.hidden_size,  # 全局 output
            bias=False,
        )
        
        # QK Norm (Qwen3 特有)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [seq_len, hidden_size]
        positions: torch.Tensor,      # [seq_len]
        freqs_cis: torch.Tensor,
        kv_cache: Optional[PagedKVCache] = None,
        request_id: str = "req_0",
        cache_start_pos: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入隐藏状态
            positions: 位置索引
            freqs_cis: RoPE 频率
            kv_cache: KV Cache 管理器
            request_id: 请求 ID
            cache_start_pos: KV Cache 写入起始位置
            is_prefill: 是否是 prefill 阶段
        """
        seq_len = hidden_states.shape[0]
        
        # =========== Step 1: QKV Projection ===========
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Reshape: [seq_len, num_heads/kv_heads, head_dim]
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim)
        
        # =========== Step 2: QK Norm ===========
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # =========== Step 3: Apply RoPE ===========
        q, k = apply_rotary_emb(q, k, freqs_cis, positions)
        
        # =========== Step 4: KV Cache 操作 ===========
        if kv_cache is not None:
            # 写入 KV Cache
            kv_cache.write_kv(request_id, k, v, cache_start_pos)
            
            # 读取完整的 KV (包括历史 tokens)
            total_seq_len = cache_start_pos + seq_len
            k_cache, v_cache = kv_cache.read_kv(request_id, total_seq_len)
        else:
            k_cache, v_cache = k, v
            total_seq_len = seq_len
        
        # =========== Step 5: Attention 计算 ===========
        if is_prefill:
            attn_output = self._prefill_attention(q, k_cache, v_cache, seq_len)
        else:
            attn_output = self._decode_attention(q, k_cache, v_cache)
        
        # =========== Step 6: Output Projection ===========
        output = self.o_proj(attn_output.reshape(seq_len, -1))
        
        return output
    
    def _prefill_attention(
        self,
        q: torch.Tensor,  # [seq_len, num_heads, head_dim]
        k: torch.Tensor,  # [total_seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [total_seq_len, num_kv_heads, head_dim]
        query_len: int,
    ) -> torch.Tensor:
        """
        Prefill 阶段的 Attention 计算
        
        特点:
        - 所有 query tokens 同时计算
        - 使用 causal mask
        - 可以利用 FlashAttention 等优化 kernel
        """
        total_seq_len = k.shape[0]
        
        # GQA: 扩展 K, V 到与 Q 相同的 heads 数
        num_kv_groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_kv_groups, dim=1)  # [total_seq, num_heads, head_dim]
        v = v.repeat_interleave(num_kv_groups, dim=1)
        
        # 只关注最后 query_len 个位置 (chunked prefill 场景)
        q_start = total_seq_len - query_len
        
        # 计算 attention scores
        # Q: [query_len, num_heads, head_dim]
        # K: [total_seq_len, num_heads, head_dim]
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scaling
        # scores: [num_heads, query_len, total_seq_len]
        
        # Causal mask: 每个 query 只能看到它之前的 keys (包括自己)
        causal_mask = torch.triu(
            torch.full((query_len, total_seq_len), float("-inf"), device=q.device),
            diagonal=q_start + 1,
        )
        scores = scores + causal_mask.unsqueeze(0)
        
        # Softmax (转换回原始 dtype)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        
        # Attention output
        # attn_weights: [num_heads, query_len, total_seq_len]
        # V: [total_seq_len, num_heads, head_dim]
        attn_output = torch.einsum("hqk,khd->qhd", attn_weights, v)
        
        return attn_output
    
    def _decode_attention(
        self,
        q: torch.Tensor,  # [1, num_heads, head_dim]
        k: torch.Tensor,  # [total_seq_len, num_kv_heads, head_dim]
        v: torch.Tensor,  # [total_seq_len, num_kv_heads, head_dim]
    ) -> torch.Tensor:
        """
        Decode 阶段的 Attention 计算
        
        特点:
        - 每次只处理 1 个新 token
        - 无需 causal mask (因为只有一个 query)
        - Memory-bound，需要读取完整 KV Cache
        """
        # GQA: 扩展 K, V
        num_kv_groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
        
        # 计算 attention scores
        # Q: [1, num_heads, head_dim]
        # K: [total_seq_len, num_heads, head_dim]
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scaling
        # scores: [num_heads, 1, total_seq_len]
        
        # Softmax (无需 mask，转换回原始 dtype)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
        
        # Attention output
        attn_output = torch.einsum("hqk,khd->qhd", attn_weights, v)
        
        return attn_output


# ============================================================================
# 5. MLP 层
# ============================================================================

class Qwen3MLP(nn.Module):
    """
    Qwen3 的 MLP 层
    
    结构: SiLU(gate_proj(x)) * up_proj(x) -> down_proj
    支持 Tensor Parallelism
    """
    
    def __init__(self, config: Qwen3Config, tp_rank: int = 0):
        super().__init__()
        
        self.tp_rank = tp_rank
        self.tp_size = config.tp_size
        
        # TP 分片: intermediate_size 在 tp_size 上分片
        self.intermediate_size_per_tp = config.intermediate_size // config.tp_size
        
        # Gate 和 Up 是 Column Parallel
        self.gate_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size_per_tp,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            self.intermediate_size_per_tp,
            bias=False,
        )
        
        # Down 是 Row Parallel
        self.down_proj = nn.Linear(
            self.intermediate_size_per_tp,
            config.hidden_size,
            bias=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================================
# 6. Decoder Layer
# ============================================================================

class Qwen3DecoderLayer(nn.Module):
    """完整的 Decoder Layer"""
    
    def __init__(self, config: Qwen3Config, layer_idx: int, tp_rank: int = 0):
        super().__init__()
        
        self.self_attn = Qwen3Attention(config, layer_idx, tp_rank)
        self.mlp = Qwen3MLP(config, tp_rank)
        
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[PagedKVCache] = None,
        request_id: str = "req_0",
        cache_start_pos: int = 0,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        # Self Attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positions,
            freqs_cis,
            kv_cache,
            request_id,
            cache_start_pos,
            is_prefill,
        )
        hidden_states = residual + hidden_states
        
        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# ============================================================================
# 7. 主演示函数
# ============================================================================

def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num: int, title: str):
    print(f"\n[Step {step_num}] {title}")
    print("-" * 50)


def demo_prefill_and_decode():
    """
    完整演示 Prefill 和 Decode 流程
    """
    # =========== 配置 ===========
    # 检测 GPU
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16  # GPU 使用 fp16 更快
        # 打印 GPU 信息
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n🚀 检测到 GPU: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
    else:
        device = "cpu"
        dtype = torch.float32  # CPU 使用 fp32
        print("\n⚠️  未检测到 GPU，使用 CPU 运行 (性能会较慢)")
        print("   提示: 确保安装了 CUDA 版本的 PyTorch:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    config = Qwen3Config(tp_size=2)
    tp_rank = 0  # 模拟 TP rank 0
    
    print_separator("vLLM Prefill & Decode 流程演示")
    print(f"\n模型配置 (Qwen3-4B with TP={config.tp_size}):")
    print(f"  - Hidden Size: {config.hidden_size}")
    print(f"  - Attention Heads: {config.num_attention_heads} (per GPU: {config.num_heads_per_tp})")
    print(f"  - KV Heads (GQA): {config.num_key_value_heads} (per GPU: {config.num_kv_heads_per_tp})")
    print(f"  - Head Dim: {config.head_dim}")
    print(f"  - 层数: {config.num_hidden_layers}")
    print(f"  - Device: {device}, Dtype: {dtype}")
    
    # =========== 初始化组件 ===========
    print_step(1, "初始化组件")
    
    # 只演示一层 (实际模型有 36 层)
    layer = Qwen3DecoderLayer(config, layer_idx=0, tp_rank=tp_rank).to(device, dtype)
    
    # KV Cache
    block_size = 16
    num_blocks = 100
    kv_cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_kv_heads_per_tp,
        head_dim=config.head_dim,
        dtype=dtype,
        device=device,
    )
    
    # RoPE
    freqs_cis = precompute_freqs_cis(
        config.head_dim,
        config.max_position_embeddings,
        device=device,
    )
    
    print(f"  KV Cache: {num_blocks} blocks × {block_size} tokens/block")
    print(f"  每个 block 内存: K + V = 2 × {config.num_kv_heads_per_tp} × {block_size} × {config.head_dim} × 2 bytes")
    print(f"  总 KV Cache 内存: {num_blocks * 2 * config.num_kv_heads_per_tp * block_size * config.head_dim * 2 / 1024 / 1024:.2f} MB")
    
    # =========== 模拟输入 ===========
    prompt_length = 128  # prompt 长度
    max_new_tokens = 10  # 生成 token 数
    request_id = "req_001"
    
    # 为请求分配 KV Cache blocks
    total_len = prompt_length + max_new_tokens
    kv_cache.allocate_blocks(request_id, total_len)
    print(f"\n  分配了 {len(kv_cache.block_tables[request_id])} 个 blocks 给请求 {request_id}")
    
    # =========== PREFILL 阶段 ===========
    print_separator("PREFILL 阶段")
    print("处理完整 prompt，生成初始 KV Cache")
    
    # 模拟 embedding 输出
    prompt_hidden = torch.randn(
        prompt_length, config.hidden_size,
        dtype=dtype, device=device
    )
    prompt_positions = torch.arange(prompt_length, device=device)
    
    print_step(2, f"Prefill: 处理 {prompt_length} 个 prompt tokens")
    print(f"  输入形状: hidden_states {list(prompt_hidden.shape)}")
    print(f"  位置范围: 0 ~ {prompt_length - 1}")
    
    # GPU Warmup (第一次运行会触发 JIT 编译，不计入时间)
    import time
    if device == "cuda":
        print("  [GPU Warmup 中...]")
        with torch.no_grad():
            _ = layer(
                hidden_states=prompt_hidden,
                positions=prompt_positions,
                freqs_cis=freqs_cis,
                kv_cache=None,  # warmup 不使用 cache
                request_id="warmup",
                cache_start_pos=0,
                is_prefill=True,
            )
        torch.cuda.synchronize()
    
    # Prefill 计算
    start = time.perf_counter()
    
    with torch.no_grad():
        prefill_output = layer(
            hidden_states=prompt_hidden,
            positions=prompt_positions,
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            request_id=request_id,
            cache_start_pos=0,
            is_prefill=True,
        )
    
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_time = time.perf_counter() - start
    
    print(f"  输出形状: {list(prefill_output.shape)}")
    print(f"  Prefill 时间: {prefill_time * 1000:.2f} ms")
    print(f"  吞吐量: {prompt_length / prefill_time:.0f} tokens/s")
    
    # 计算 FLOPs
    # QKV + O projections
    qkvo_flops = 2 * prompt_length * config.hidden_size * (
        config.num_heads_per_tp * config.head_dim +  # Q
        2 * config.num_kv_heads_per_tp * config.head_dim +  # K, V
        config.num_heads_per_tp * config.head_dim  # O
    )
    # Attention
    attn_flops = 2 * prompt_length * prompt_length * config.num_heads_per_tp * config.head_dim
    # MLP
    mlp_flops = 2 * prompt_length * config.hidden_size * 3 * (config.intermediate_size // config.tp_size)
    
    total_flops = qkvo_flops + attn_flops + mlp_flops
    print(f"  单层 FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    
    # =========== Chunked Prefill 演示 ===========
    print_separator("CHUNKED PREFILL 演示")
    print("将长 prompt 分成多个 chunk 处理")
    
    long_prompt_length = 512
    chunk_size = 128
    num_chunks = math.ceil(long_prompt_length / chunk_size)
    
    # 重新初始化请求 - 为长 prompt 分配足够的 blocks
    kv_cache.free(request_id)
    kv_cache.allocate_blocks(request_id, long_prompt_length + max_new_tokens)
    
    print(f"\n  长 Prompt: {long_prompt_length} tokens")
    print(f"  Chunk 大小: {chunk_size} tokens")
    print(f"  Chunk 数量: {num_chunks}")
    
    long_prompt_hidden = torch.randn(
        long_prompt_length, config.hidden_size,
        dtype=dtype, device=device
    )
    
    total_chunked_time = 0
    for chunk_idx in range(num_chunks):
        start_pos = chunk_idx * chunk_size
        end_pos = min(start_pos + chunk_size, long_prompt_length)
        chunk_len = end_pos - start_pos
        
        chunk_hidden = long_prompt_hidden[start_pos:end_pos]
        chunk_positions = torch.arange(start_pos, end_pos, device=device)
        
        print_step(3 + chunk_idx, f"Chunk {chunk_idx + 1}/{num_chunks}: tokens [{start_pos}, {end_pos})")
        
        start = time.perf_counter()
        with torch.no_grad():
            chunk_output = layer(
                hidden_states=chunk_hidden,
                positions=chunk_positions,
                freqs_cis=freqs_cis,
                kv_cache=kv_cache,
                request_id=request_id,
                cache_start_pos=start_pos,
                is_prefill=True,  # 仍然是 prefill 模式
            )
        if device == "cuda":
            torch.cuda.synchronize()
        chunk_time = time.perf_counter() - start
        total_chunked_time += chunk_time
        
        print(f"    处理 {chunk_len} tokens，耗时 {chunk_time * 1000:.2f} ms")
        print(f"    累计已缓存 KV: {end_pos} tokens")
    
    print(f"\n  Chunked Prefill 总时间: {total_chunked_time * 1000:.2f} ms")
    
    # =========== DECODE 阶段 ===========
    print_separator("DECODE 阶段")
    print("逐 token 自回归生成")
    
    # 重置为短 prompt
    kv_cache.free(request_id)
    kv_cache.allocate_blocks(request_id, prompt_length + max_new_tokens)
    
    # 先做一次 prefill
    with torch.no_grad():
        _ = layer(
            hidden_states=prompt_hidden,
            positions=prompt_positions,
            freqs_cis=freqs_cis,
            kv_cache=kv_cache,
            request_id=request_id,
            cache_start_pos=0,
            is_prefill=True,
        )
    
    current_pos = prompt_length
    decode_times = []
    
    for step in range(max_new_tokens):
        print_step(3 + num_chunks + step, f"Decode step {step + 1}/{max_new_tokens}")
        
        # 模拟新生成的 token 的 embedding
        new_token_hidden = torch.randn(
            1, config.hidden_size,
            dtype=dtype, device=device
        )
        new_position = torch.tensor([current_pos], device=device)
        
        start = time.perf_counter()
        with torch.no_grad():
            decode_output = layer(
                hidden_states=new_token_hidden,
                positions=new_position,
                freqs_cis=freqs_cis,
                kv_cache=kv_cache,
                request_id=request_id,
                cache_start_pos=current_pos,
                is_prefill=False,  # Decode 模式
            )
        if device == "cuda":
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - start
        decode_times.append(decode_time * 1000)
        
        print(f"    位置: {current_pos}")
        print(f"    需要读取 KV Cache: {current_pos + 1} tokens")
        print(f"    耗时: {decode_time * 1000:.2f} ms")
        
        current_pos += 1
    
    # =========== 统计信息 ===========
    print_separator("统计信息")
    
    avg_decode_time = sum(decode_times) / len(decode_times)
    print(f"\n  Prefill ({prompt_length} tokens):")
    print(f"    时间: {prefill_time * 1000:.2f} ms")
    print(f"    吞吐量: {prompt_length / prefill_time:.0f} tokens/s")
    print(f"    特点: Compute-bound (矩阵乘法密集)")
    
    print(f"\n  Decode ({max_new_tokens} tokens):")
    print(f"    平均每 token: {avg_decode_time:.2f} ms")
    print(f"    吞吐量: {1000 / avg_decode_time:.0f} tokens/s")
    print(f"    特点: Memory-bound (KV Cache 读取)")
    
    # KV Cache 内存计算
    kv_cache_bytes_per_token = 2 * config.num_kv_heads_per_tp * config.head_dim * 2  # K + V, bf16
    print(f"\n  KV Cache 内存 (per token, per layer, per GPU):")
    print(f"    {kv_cache_bytes_per_token} bytes = 2 (K+V) × {config.num_kv_heads_per_tp} heads × {config.head_dim} dim × 2 bytes")
    print(f"    全模型 ({config.num_hidden_layers} 层): {kv_cache_bytes_per_token * config.num_hidden_layers / 1024:.1f} KB/token")
    
    # 计算对比
    print(f"\n  Prefill vs Decode 特性对比:")
    print(f"    {'特性':<20} {'Prefill':<25} {'Decode':<25}")
    print(f"    {'-'*70}")
    print(f"    {'Query tokens':<20} {prompt_length:<25} {'1':<25}")
    print(f"    {'KV tokens':<20} {prompt_length:<25} {f'1~{prompt_length + max_new_tokens}':<25}")
    print(f"    {'Attention 复杂度':<20} {'O(seq_len²)':<25} {'O(seq_len)':<25}")
    print(f"    {'瓶颈':<20} {'Compute':<25} {'Memory Bandwidth':<25}")
    print(f"    {'Batching 收益':<20} {'高 (更多并行)':<25} {'中等 (受限于内存)':<25}")
    
    # GPU 内存使用情况
    if device == "cuda":
        print(f"\n  GPU 内存使用:")
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"    已分配: {allocated:.1f} MB")
        print(f"    已保留: {reserved:.1f} MB")
    
    print("\n" + "=" * 70)
    print("  演示完成!")
    print("=" * 70)


# ============================================================================
# 8. 运行演示
# ============================================================================

if __name__ == "__main__":
    demo_prefill_and_decode()
