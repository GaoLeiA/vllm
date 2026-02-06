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
PATCHES = {
    # =========================================================================
    # Patch 1: qwen3.py - 添加 Attention 初始化和 forward 日志
    # =========================================================================
    "model_executor/models/qwen3.py": {
        "backup": True,
        "insertions": [
            {
                "after_line": "from vllm.model_executor.layers.vocab_parallel_embedding import (",
                "content": """
# ===== DEBUG LOGGING =====
import logging
_debug_logger = logging.getLogger("vllm.study.qwen3")
_debug_logger.setLevel(logging.DEBUG)
if not _debug_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s | QWEN3 | %(message)s', datefmt='%H:%M:%S'))
    _debug_logger.addHandler(_handler)
# ===== END DEBUG LOGGING =====
"""
            },
        ],
        "replacements": [
            {
                "original": """    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size""",
                "replacement": """    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # ===== DEBUG: Log attention head configuration =====
        from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        _debug_logger.info(
            f"[TP_RANK={tp_rank}] 🔧 Qwen3Attention.__init__: "
            f"hidden={hidden_size}, heads={num_heads}(total)->分片后将计算, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}, prefix={prefix}"
        )
        # ===== END DEBUG ====="""
            }
        ]
    },
    
    # =========================================================================
    # Patch 2: parallel_state.py - 添加 AllReduce 日志
    # =========================================================================
    "distributed/parallel_state.py": {
        "backup": True,
        "insertions": [],
        "replacements": [
            {
                "original": """    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        \"\"\"
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        \"\"\"
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_""",
                "replacement": """    _allreduce_counter = 0  # Class-level counter for debugging
    
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        \"\"\"
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        \"\"\"
        # ===== DEBUG: Log AllReduce operations =====
        GroupCoordinator._allreduce_counter += 1
        count = GroupCoordinator._allreduce_counter
        if count <= 10 or count % 200 == 0:
            logger.debug(
                f"📡 AllReduce #{count}: shape={input_.shape}, "
                f"rank={self.rank_in_group}/{self.world_size}, group={self.unique_name}"
            )
        # ===== END DEBUG =====
        
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_"""
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
