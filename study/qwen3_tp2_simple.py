#!/usr/bin/env python3
"""
Qwen3 TP=2 简化推理脚本
=======================

这是一个简化版本的推理脚本，专注于验证 TP=2 配置能正常工作。
vLLM 本身的日志已经提供了足够的信息来观察关键行为。

使用方法:
---------
# 设置环境变量启用详细日志
export VLLM_LOGGING_LEVEL=DEBUG

# 运行脚本
python study/qwen3_tp2_simple.py

观察重点:
---------
1. Worker 进程日志中的 "Qwen3Attention" 或 "QKVParallelLinear" 初始化信息
2. "Loading weights" 过程中的分片信息
3. AllReduce 通信（在高级日志中可见）
"""

import os
import time

# 启用详细日志
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

def main():
    import torch
    from vllm import LLM, SamplingParams
    
    print("=" * 70)
    print("🎯 Qwen3 TP=2 推理测试")
    print("=" * 70)
    
    # 配置
    # 如果你有 Qwen3-8B，可以改成 "Qwen/Qwen3-8B"
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    TENSOR_PARALLEL_SIZE = 2
    
    # 检查 GPU
    gpu_count = torch.cuda.device_count()
    print(f"\n📊 GPU 信息:")
    print(f"  可用 GPU 数量: {gpu_count}")
    
    if gpu_count < TENSOR_PARALLEL_SIZE:
        print(f"  ❌ GPU 不足! 需要 {TENSOR_PARALLEL_SIZE} 个")
        print(f"  💡 设置 TENSOR_PARALLEL_SIZE = 1 来使用单 GPU")
        TENSOR_PARALLEL_SIZE = min(gpu_count, 1)
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")
    
    # 初始化 LLM
    print(f"\n📦 加载模型: {MODEL_NAME}")
    print(f"   Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    print("   (请观察下方 Worker 日志中的分片信息...)")
    print("-" * 70)
    
    start_load = time.time()
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    load_time = time.time() - start_load
    
    print("-" * 70)
    print(f"✅ 模型加载完成! 耗时: {load_time:.2f} 秒")
    
    # 打印配置摘要
    try:
        vllm_config = llm.llm_engine.vllm_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        
        # 获取模型配置（注意：get_num_layers 需要 parallel_config 参数）
        hidden_size = model_config.get_hidden_size()
        num_layers = model_config.get_num_layers(parallel_config)
        num_heads = model_config.get_num_attention_heads()
        num_kv_heads = model_config.get_num_kv_heads()
        tp_size = parallel_config.tensor_parallel_size
        
        print("\n📊 模型配置:")
        print(f"  模型: {model_config.model}")
        print(f"  隐藏层维度: {hidden_size}")
        print(f"  层数: {num_layers}")
        print(f"  注意力头数: {num_heads}")
        print(f"  KV 头数: {num_kv_heads}")
        
        print(f"\n📊 并行配置:")
        print(f"  Tensor Parallel: {tp_size}")
        print(f"  Pipeline Parallel: {parallel_config.pipeline_parallel_size}")
        
        # 计算分片后的配置
        print(f"\n📊 每个 GPU 上的分片:")
        print(f"  注意力头数/GPU: {num_heads} / {tp_size} = {num_heads // tp_size}")
        print(f"  KV 头数/GPU: {num_kv_heads} / {tp_size} = {num_kv_heads // tp_size}")
        
    except Exception as e:
        print(f"  (无法读取详细配置: {e})")
    
    # 运行推理
    print("\n" + "=" * 70)
    print("🚀 开始推理测试")
    print("=" * 70)
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "In machine learning, tensor parallelism means",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=30,
        top_p=0.9,
    )
    
    print(f"\n采样参数: temperature={sampling_params.temperature}, max_tokens={sampling_params.max_tokens}")
    
    start_infer = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - start_infer
    
    # 打印结果
    print(f"\n推理耗时: {infer_time:.2f} 秒")
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"生成 token 总数: {total_tokens}")
    print(f"Throughput: {total_tokens / infer_time:.1f} tokens/s")
    
    print("\n" + "-" * 70)
    for i, output in enumerate(outputs):
        print(f"\n📝 输出 #{i+1}:")
        print(f"   Prompt: {output.prompt[:50]}...")
        print(f"   Generated: {output.outputs[0].text}")
        print(f"   Tokens: {len(output.outputs[0].token_ids)}")
    
    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)
    
    # 提示观察点
    print("""
💡 观察重点:
   1. 在上方 Worker 日志中，查找包含 "Loading" 的行，观察权重加载过程
   2. 如果设置 VLLM_LOGGING_LEVEL=DEBUG，可以看到更多分片和通信信息
   3. 多 GPU 时，每个 Worker 进程 (Worker_TP0, Worker_TP1) 会有独立的日志

📚 深入学习:
   - 查看 study/Qwen3_TP2_Inference_Explained.md 了解详细原理
   - 使用 python study/apply_debug_patch.py --apply 注入更详细的日志
""")


if __name__ == "__main__":
    main()
