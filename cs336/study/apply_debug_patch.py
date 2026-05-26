#!/usr/bin/env python3
"""
vLLM 日志注入补丁
=================

这个脚本会临时修改 vLLM 源码，添加详细的调试日志。
运行完毕后可以使用 --restore 参数恢复原始代码。

使用方法:
---------
# 应用补丁（添加日志）
python study/apply_debug_patch.py --apply

# 恢复原始代码
python study/apply_debug_patch.py --restore

# 运行带日志的推理
python study/qwen3_tp2_inference_debug.py
"""

import argparse
import shutil
import os
from pathlib import Path

# vLLM 源码根目录
VLLM_ROOT = Path(__file__).parent.parent / "vllm"

# 需要修改的文件及其补丁
# 注意：避免在会被 torch.compile 追踪的代码路径中添加动态属性修改！
PATCHES = {
    # =========================================================================
    # Patch 1: qwen2.py (Qwen2.5 使用 qwen2.py) - 在初始化时添加日志
    # 这是安全的，因为 __init__ 不会被 torch.compile 追踪
    # =========================================================================
    "model_executor/models/qwen2.py": {
        "backup": True,
        "insertions": [],
        "replacements": [
            {
                # 在 Qwen2Attention.__init__ 末尾添加日志
                "original": """        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )""",
                "replacement": """        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        
        # ===== DEBUG: 打印 Attention 初始化信息 =====
        from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        logger.info(
            f"[TP_RANK={tp_rank}/{tp_size}] Qwen2Attention.__init__: "
            f"hidden={self.hidden_size}, total_heads={self.total_num_heads}, "
            f"heads/GPU={self.num_heads}, total_kv_heads={self.total_num_kv_heads}, "
            f"kv_heads/GPU={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"q_size={self.q_size}, kv_size={self.kv_size}, prefix={prefix}"
        )
        # ===== END DEBUG ====="""
            }
        ]
    },
    
    # =========================================================================
    # Patch 2: linear.py - 在 QKVParallelLinear 初始化时添加日志
    # =========================================================================
    "model_executor/layers/linear.py": {
        "backup": True,
        "insertions": [],
        "replacements": [
            {
                "original": """        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def _get_shard_offset_mapping(self, loaded_shard_id: str):""",
                "replacement": """        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )
        
        # ===== DEBUG: 打印 QKV 分片信息 =====
        logger.info(
            f"[TP={tp_size}] QKVParallelLinear: prefix={prefix}, "
            f"hidden={self.hidden_size}, head_size={self.head_size}, "
            f"total_heads={self.total_num_heads}, heads/GPU={self.num_heads}, "
            f"total_kv_heads={self.total_num_kv_heads}, kv_heads/GPU={self.num_kv_heads}, "
            f"output_sizes={self.output_sizes}"
        )
        # ===== END DEBUG =====

    def _get_shard_offset_mapping(self, loaded_shard_id: str):"""
            }
        ]
    },
}

def apply_patch(file_path: Path, patch_config: dict) -> bool:
    """应用单个文件的补丁"""
    full_path = VLLM_ROOT / file_path
    backup_path = full_path.with_suffix(full_path.suffix + ".bak")
    
    if not full_path.exists():
        print(f"❌ 文件不存在: {full_path}")
        return False
    
    # 备份原文件
    if patch_config.get("backup", True) and not backup_path.exists():
        shutil.copy2(full_path, backup_path)
        print(f"📁 已备份: {backup_path}")
    
    # 读取文件内容
    content = full_path.read_text(encoding="utf-8")
    original_content = content
    
    # 应用插入
    for insertion in patch_config.get("insertions", []):
        after_line = insertion["after_line"]
        new_content = insertion["content"]
        if after_line in content and new_content not in content:
            content = content.replace(after_line, after_line + new_content)
            print(f"  ✅ 插入成功: after '{after_line[:50]}...'")
    
    # 应用替换
    for replacement in patch_config.get("replacements", []):
        original = replacement["original"]
        new = replacement["replacement"]
        if original in content:
            content = content.replace(original, new)
            print(f"  ✅ 替换成功: '{original[:50]}...'")
        elif new in content:
            print(f"  ⏭️ 已应用: '{original[:50]}...'")
        else:
            print(f"  ⚠️ 未找到: '{original[:50]}...'")
    
    # 写入修改后的内容
    if content != original_content:
        full_path.write_text(content, encoding="utf-8")
        print(f"✅ 已修改: {file_path}")
        return True
    else:
        print(f"⏭️ 无需修改: {file_path}")
        return False


def restore_file(file_path: Path) -> bool:
    """恢复单个文件"""
    full_path = VLLM_ROOT / file_path
    backup_path = full_path.with_suffix(full_path.suffix + ".bak")
    
    if backup_path.exists():
        shutil.copy2(backup_path, full_path)
        backup_path.unlink()
        print(f"✅ 已恢复: {file_path}")
        return True
    else:
        print(f"⚠️ 无备份可恢复: {file_path}")
        return False


def main():
    parser = argparse.ArgumentParser(description="vLLM 调试补丁管理工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="应用调试日志补丁")
    group.add_argument("--restore", action="store_true", help="恢复原始代码")
    group.add_argument("--status", action="store_true", help="检查补丁状态")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("vLLM 调试补丁管理工具")
    print("=" * 60)
    print(f"vLLM 路径: {VLLM_ROOT}")
    print()
    
    if args.apply:
        print("📝 应用调试日志补丁...")
        for file_path, patch_config in PATCHES.items():
            print(f"\n处理文件: {file_path}")
            apply_patch(Path(file_path), patch_config)
        
        print("\n" + "=" * 60)
        print("✅ 补丁应用完成！")
        print("现在可以运行: python study/qwen3_tp2_inference_debug.py")
        print("完成后使用: python study/apply_debug_patch.py --restore 恢复代码")
        
    elif args.restore:
        print("🔄 恢复原始代码...")
        for file_path in PATCHES.keys():
            restore_file(Path(file_path))
        
        print("\n" + "=" * 60)
        print("✅ 代码已恢复！")
        
    elif args.status:
        print("📊 检查补丁状态...")
        for file_path in PATCHES.keys():
            full_path = VLLM_ROOT / file_path
            backup_path = full_path.with_suffix(full_path.suffix + ".bak")
            
            status = "🔵 已应用补丁" if backup_path.exists() else "⚪ 原始状态"
            print(f"  {file_path}: {status}")


if __name__ == "__main__":
    main()
