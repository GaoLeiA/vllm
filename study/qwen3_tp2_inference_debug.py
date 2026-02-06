#!/usr/bin/env python3
"""
Qwen3 8B 模型推理调试脚本 (TP=2)
==================================

本脚本用于观察 vLLM 中 Qwen3 8B 模型在 Tensor Parallelism=2 配置下的推理过程。
通过注入详细的日志，你可以观察到：

1. 模型加载过程
2. 张量并行分片策略
3. 注意力计算流程
4. 分布式通信操作 (AllReduce, AllGather)
5. 推理执行流程

使用方法:
---------
# 在 WSL/Linux 环境下运行 (需要2个GPU)
python study/qwen3_tp2_inference_debug.py

# 或者使用 CUDA_VISIBLE_DEVICES 指定显卡
CUDA_VISIBLE_DEVICES=0,1 python study/qwen3_tp2_inference_debug.py

环境要求:
---------
- 2张 GPU (每张至少 20GB 显存，用于 8B 模型)
- 安装好 vLLM
- Qwen3 模型权重 (可以使用 HuggingFace 模型ID)

注意: 如果显存不足，可以改用较小的模型如 Qwen2.5-1.5B 进行测试
"""

import os
import sys
import logging
import functools
from typing import Optional
import time

# 设置日志级别 - 在导入 vLLM 之前设置
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# ============================================================================
# 第一部分：日志配置
# ============================================================================

def setup_study_logger():
    """设置专门用于学习的日志记录器"""
    logger = logging.getLogger("vllm_study")
    logger.setLevel(logging.DEBUG)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # 格式化器 - 包含时间、rank信息、模块名等
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（可选）
    file_handler = logging.FileHandler('study/qwen3_tp2_debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

study_logger = setup_study_logger()

# ============================================================================
# 第二部分：Monkey Patch - 注入观察日志
# ============================================================================

def inject_logging_hooks():
    """
    注入日志钩子到 vLLM 的关键模块
    
    这些钩子会在关键操作时打印详细信息，帮助你理解:
    1. 模型初始化时的张量并行分片
    2. 前向传播时的数据流动
    3. 分布式通信操作
    """
    
    # -------------------------------------------------------------------------
    # Hook 1: Qwen3Attention 初始化 - 观察注意力头的分片
    # -------------------------------------------------------------------------
    try:
        from vllm.model_executor.models import qwen3
        original_qwen3_attention_init = qwen3.Qwen3Attention.__init__
        
        @functools.wraps(original_qwen3_attention_init)
        def patched_qwen3_attention_init(self, *args, **kwargs):
            result = original_qwen3_attention_init(self, *args, **kwargs)
            
            # 获取 TP 相关信息
            from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            
            study_logger.info(
                f"[TP_RANK={tp_rank}] 🔧 Qwen3Attention 初始化:\n"
                f"    ├── hidden_size: {self.hidden_size}\n"
                f"    ├── total_num_heads: {self.total_num_heads} (全部注意力头数)\n"
                f"    ├── num_heads (本分片): {self.num_heads} (= {self.total_num_heads} / {tp_size})\n"
                f"    ├── total_num_kv_heads: {self.total_num_kv_heads} (全部KV头数)\n"
                f"    ├── num_kv_heads (本分片): {self.num_kv_heads}\n"
                f"    ├── head_dim: {self.head_dim}\n"
                f"    ├── q_size (本分片): {self.q_size} (= {self.num_heads} * {self.head_dim})\n"
                f"    └── kv_size (本分片): {self.kv_size} (= {self.num_kv_heads} * {self.head_dim})"
            )
            return result
        
        qwen3.Qwen3Attention.__init__ = patched_qwen3_attention_init
        study_logger.info("✅ Hook 注入成功: Qwen3Attention.__init__")
    except Exception as e:
        study_logger.warning(f"⚠️ Hook 注入失败 (Qwen3Attention): {e}")
    
    # -------------------------------------------------------------------------
    # Hook 2: Qwen3Attention.forward - 观察注意力计算
    # -------------------------------------------------------------------------
    try:
        from vllm.model_executor.models import qwen3
        original_qwen3_attention_forward = qwen3.Qwen3Attention.forward
        
        @functools.wraps(original_qwen3_attention_forward)
        def patched_qwen3_attention_forward(self, positions, hidden_states):
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            
            # 只在第一个 token 时打印，避免日志过多
            if hidden_states.shape[0] <= 32:  # 只对小批量打印
                study_logger.debug(
                    f"[TP_RANK={tp_rank}] 🔄 Qwen3Attention.forward:\n"
                    f"    ├── input hidden_states: {hidden_states.shape} {hidden_states.dtype}\n"
                    f"    ├── positions: {positions.shape}"
                )
            
            output = original_qwen3_attention_forward(self, positions, hidden_states)
            
            if hidden_states.shape[0] <= 32:
                study_logger.debug(
                    f"[TP_RANK={tp_rank}] ✅ Qwen3Attention.forward 完成:\n"
                    f"    └── output: {output.shape}"
                )
            return output
        
        qwen3.Qwen3Attention.forward = patched_qwen3_attention_forward
        study_logger.info("✅ Hook 注入成功: Qwen3Attention.forward")
    except Exception as e:
        study_logger.warning(f"⚠️ Hook 注入失败 (Qwen3Attention.forward): {e}")
    
    # -------------------------------------------------------------------------
    # Hook 3: QKVParallelLinear 初始化 - 观察 QKV 权重分片
    # -------------------------------------------------------------------------
    try:
        from vllm.model_executor.layers import linear
        original_qkv_init = linear.QKVParallelLinear.__init__
        
        @functools.wraps(original_qkv_init)
        def patched_qkv_init(self, hidden_size, head_size, total_num_heads, 
                             total_num_kv_heads=None, *args, **kwargs):
            result = original_qkv_init(self, hidden_size, head_size, total_num_heads,
                                       total_num_kv_heads, *args, **kwargs)
            
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            
            study_logger.info(
                f"[TP_RANK={tp_rank}] 🔧 QKVParallelLinear 初始化:\n"
                f"    ├── hidden_size: {self.hidden_size}\n"
                f"    ├── head_size: {self.head_size}\n"
                f"    ├── total_num_heads: {self.total_num_heads}\n"
                f"    ├── num_heads (本分片): {self.num_heads}\n"
                f"    ├── total_num_kv_heads: {self.total_num_kv_heads}\n"
                f"    ├── num_kv_heads (本分片): {self.num_kv_heads}\n"
                f"    ├── num_kv_head_replicas: {self.num_kv_head_replicas}\n"
                f"    └── output_sizes: {self.output_sizes}"
            )
            return result
        
        linear.QKVParallelLinear.__init__ = patched_qkv_init
        study_logger.info("✅ Hook 注入成功: QKVParallelLinear.__init__")
    except Exception as e:
        study_logger.warning(f"⚠️ Hook 注入失败 (QKVParallelLinear): {e}")
    
    # -------------------------------------------------------------------------
    # Hook 4: RowParallelLinear.forward - 观察 AllReduce 操作
    # -------------------------------------------------------------------------
    try:
        from vllm.model_executor.layers import linear
        
        # 找到 RowParallelLinear 的 forward 方法 (通过 CustomOp)
        # 这里我们 hook quant_method.apply
        original_row_forward = linear.RowParallelLinear.forward_native
        
        @functools.wraps(original_row_forward)
        def patched_row_forward(self, input_):
            from vllm.distributed import get_tensor_model_parallel_rank
            tp_rank = get_tensor_model_parallel_rank()
            
            # 打印输入信息
            if input_.shape[0] <= 32:
                study_logger.debug(
                    f"[TP_RANK={tp_rank}] 📡 RowParallelLinear.forward (含 AllReduce):\n"
                    f"    ├── input: {input_.shape}\n"
                    f"    ├── input_size_per_partition: {self.input_size_per_partition}\n"
                    f"    └── reduce_results: {self.reduce_results}"
                )
            
            output = original_row_forward(self, input_)
            
            if input_.shape[0] <= 32 and self.reduce_results:
                study_logger.debug(
                    f"[TP_RANK={tp_rank}] ✅ RowParallelLinear AllReduce 完成\n"
                    f"    └── output shape: {output[0].shape if isinstance(output, tuple) else output.shape}"
                )
            
            return output
        
        linear.RowParallelLinear.forward_native = patched_row_forward
        study_logger.info("✅ Hook 注入成功: RowParallelLinear.forward_native")
    except Exception as e:
        study_logger.warning(f"⚠️ Hook 注入失败 (RowParallelLinear): {e}")
    
    # -------------------------------------------------------------------------
    # Hook 5: GroupCoordinator.all_reduce - 观察实际的 AllReduce 通信
    # -------------------------------------------------------------------------
    try:
        from vllm.distributed import parallel_state
        original_all_reduce = parallel_state.GroupCoordinator.all_reduce
        
        call_count = [0]  # 使用列表来避免 closure 问题
        
        @functools.wraps(original_all_reduce)
        def patched_all_reduce(self, input_):
            call_count[0] += 1
            
            # 每100次打印一次，避免日志过多
            if call_count[0] <= 20 or call_count[0] % 100 == 0:
                study_logger.debug(
                    f"📡 AllReduce #{call_count[0]}:\n"
                    f"    ├── input shape: {input_.shape}\n"
                    f"    ├── world_size: {self.world_size}\n"
                    f"    ├── rank: {self.rank_in_group}/{self.world_size}\n"
                    f"    └── group: {self.unique_name}"
                )
            
            result = original_all_reduce(self, input_)
            return result
        
        parallel_state.GroupCoordinator.all_reduce = patched_all_reduce
        study_logger.info("✅ Hook 注入成功: GroupCoordinator.all_reduce")
    except Exception as e:
        study_logger.warning(f"⚠️ Hook 注入失败 (GroupCoordinator.all_reduce): {e}")


# ============================================================================
# 第三部分：模型配置和推理
# ============================================================================

def print_model_summary(llm):
    """打印模型配置摘要"""
    try:
        # vLLM v1 API: 使用 vllm_config
        vllm_config = llm.llm_engine.vllm_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        
        # 获取模型配置（注意：某些方法需要 parallel_config 参数）
        hidden_size = model_config.get_hidden_size()
        num_layers = model_config.get_num_layers(parallel_config)
        num_attention_heads = model_config.get_num_attention_heads()
        num_kv_heads = model_config.get_num_kv_heads()
        vocab_size = model_config.get_vocab_size()
        
        study_logger.info(
            f"\n{'='*70}\n"
            f"📊 模型配置摘要\n"
            f"{'='*70}\n"
            f"模型名称: {model_config.model}\n"
            f"模型架构: {model_config.architectures}\n"
            f"隐藏层维度: {hidden_size}\n"
            f"层数: {num_layers}\n"
            f"注意力头数: {num_attention_heads}\n"
            f"KV 头数: {num_kv_heads}\n"
            f"词汇表大小: {vocab_size}\n"
            f"最大序列长度: {model_config.max_model_len}\n"
            f"{'='*70}\n"
            f"📊 并行配置\n"
            f"{'='*70}\n"
            f"Tensor Parallel Size: {parallel_config.tensor_parallel_size}\n"
            f"Pipeline Parallel Size: {parallel_config.pipeline_parallel_size}\n"
            f"Data Parallel Size: {parallel_config.data_parallel_size}\n"
            f"{'='*70}\n"
            f"📊 每个 GPU 上的分片 (TP={parallel_config.tensor_parallel_size})\n"
            f"{'='*70}\n"
            f"注意力头数/GPU: {num_attention_heads} / {parallel_config.tensor_parallel_size} = {num_attention_heads // parallel_config.tensor_parallel_size}\n"
            f"KV 头数/GPU: {num_kv_heads} / {parallel_config.tensor_parallel_size} = {num_kv_heads // parallel_config.tensor_parallel_size}\n"
            f"{'='*70}"
        )
    except Exception as e:
        # 简化版输出
        study_logger.warning(f"无法获取完整模型配置: {e}")
        try:
            vllm_config = llm.llm_engine.vllm_config
            model_config = vllm_config.model_config
            study_logger.info(
                f"\n{'='*70}\n"
                f"📊 模型配置摘要 (简化版)\n"
                f"{'='*70}\n"
                f"模型名称: {model_config.model}\n"
                f"{'='*70}"
            )
        except Exception as e2:
            study_logger.warning(f"无法获取模型配置: {e2}")


def analyze_weight_distribution(llm):
    """分析权重在不同 GPU 上的分布 (v1 引擎可能不支持直接访问)"""
    study_logger.info("\n" + "="*70)
    study_logger.info("📊 权重分布分析")
    study_logger.info("="*70)
    study_logger.info(
        "💡 提示: 在 vLLM v1 架构中，模型运行在独立的 Worker 进程中，\n"
        "    主进程无法直接访问模型权重。请查看 Worker 进程的日志来观察权重分片。\n"
        "    你可以在上面的初始化日志中看到每个 TP rank 的 QKV 分片信息。"
    )


def run_inference_with_logging(llm, prompts):
    """运行推理并记录详细日志"""
    from vllm import SamplingParams
    
    study_logger.info("\n" + "="*70)
    study_logger.info("🚀 开始推理")
    study_logger.info("="*70)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        top_p=0.9,
    )
    
    study_logger.info(f"采样参数: {sampling_params}")
    study_logger.info(f"输入 prompts: {prompts}")
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    study_logger.info(f"\n推理耗时: {end_time - start_time:.2f} 秒")
    
    for i, output in enumerate(outputs):
        study_logger.info(
            f"\n{'='*50}\n"
            f"输出 #{i+1}:\n"
            f"Prompt: {output.prompt}\n"
            f"Generated: {output.outputs[0].text}\n"
            f"Token count: {len(output.outputs[0].token_ids)}\n"
            f"{'='*50}"
        )
    
    return outputs


def main():
    """主函数"""
    study_logger.info("\n" + "="*70)
    study_logger.info("🎯 vLLM Qwen3 TP=2 推理调试脚本")
    study_logger.info("="*70)
    
    # 注入日志钩子 (在导入 vLLM 之前)
    study_logger.info("📌 注入观察日志钩子...")
    inject_logging_hooks()
    
    # 导入 vLLM
    from vllm import LLM, SamplingParams
    
    # 模型配置
    # 你可以根据实际情况修改这些参数
    MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # 使用较小的模型进行测试，避免显存不足
    # MODEL_NAME = "Qwen/Qwen3-8B"    # 如果有足够显存，可以使用这个
    
    TENSOR_PARALLEL_SIZE = 2  # 使用2个GPU进行张量并行
    
    study_logger.info(f"\n加载模型: {MODEL_NAME}")
    study_logger.info(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    
    # 检查 GPU 数量
    import torch
    gpu_count = torch.cuda.device_count()
    study_logger.info(f"可用 GPU 数量: {gpu_count}")
    
    if gpu_count < TENSOR_PARALLEL_SIZE:
        study_logger.error(
            f"❌ GPU 数量不足! 需要 {TENSOR_PARALLEL_SIZE} 个, 只有 {gpu_count} 个\n"
            f"请调整 TENSOR_PARALLEL_SIZE 或使用更多 GPU"
        )
        return
    
    # 打印每个 GPU 的信息
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        study_logger.info(
            f"GPU {i}: {props.name}, "
            f"显存: {props.total_memory / 1024**3:.1f} GB"
        )
    
    # 初始化 LLM
    study_logger.info("\n" + "="*70)
    study_logger.info("📦 初始化 vLLM LLM 引擎...")
    study_logger.info("="*70)
    
    try:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            # 降低显存使用
            gpu_memory_utilization=0.85,
            max_model_len=2048,  # 限制最大长度以节省显存
        )
    except Exception as e:
        study_logger.error(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 打印模型摘要
    print_model_summary(llm)
    
    # 分析权重分布
    analyze_weight_distribution(llm)
    
    # 测试推理
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    
    outputs = run_inference_with_logging(llm, prompts)
    
    study_logger.info("\n" + "="*70)
    study_logger.info("✅ 调试完成!")
    study_logger.info("="*70)
    study_logger.info("查看完整日志: study/qwen3_tp2_debug.log")


if __name__ == "__main__":
    main()
