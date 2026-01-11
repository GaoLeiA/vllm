#!/usr/bin/env python3
"""
Continuous Batching 演示

展示 vLLM 的核心调度策略：
1. 请求动态加入和退出 batch
2. GPU 利用率最大化
3. Prefill 和 Decode 混合调度
"""

import time
import random
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

# ============================================================================
# 1. 请求和调度器数据结构
# ============================================================================

@dataclass
class Request:
    """模拟一个推理请求"""
    request_id: str
    prompt_tokens: int          # prompt 长度
    max_new_tokens: int         # 最大生成长度
    
    # 状态
    status: str = "waiting"     # waiting, prefilling, decoding, finished
    computed_tokens: int = 0    # 已计算的 token 数
    generated_tokens: int = 0   # 已生成的 token 数
    
    # 时间统计
    arrival_time: float = field(default_factory=time.time)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens
    
    @property
    def remaining_tokens(self) -> int:
        return self.max_new_tokens - self.generated_tokens
    
    @property
    def is_prefilling(self) -> bool:
        return self.computed_tokens < self.prompt_tokens
    
    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token"""
        if self.first_token_time:
            return self.first_token_time - self.arrival_time
        return None
    
    @property
    def total_time(self) -> Optional[float]:
        if self.finish_time:
            return self.finish_time - self.arrival_time
        return None


@dataclass 
class SchedulerConfig:
    """调度器配置"""
    max_num_seqs: int = 8               # 最大并发请求数
    max_num_batched_tokens: int = 2048  # 每步最大 token 数
    enable_chunked_prefill: bool = True # 启用分块 prefill
    chunk_size: int = 512               # Prefill chunk 大小


class ContinuousBatchingScheduler:
    """
    Continuous Batching 调度器
    
    核心功能:
    1. 动态管理 running batch
    2. 请求完成后立即释放资源
    3. 新请求可以随时加入
    4. Prefill 和 Decode 混合调度
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        
        # 请求队列
        self.waiting_queue: deque[Request] = deque()  # 等待队列
        self.running_batch: list[Request] = []        # 当前运行的 batch
        self.finished_requests: list[Request] = []   # 已完成的请求
        
        # 统计
        self.step_count = 0
        self.total_tokens_processed = 0
    
    def add_request(self, request: Request):
        """添加新请求到等待队列"""
        request.status = "waiting"
        self.waiting_queue.append(request)
        print(f"  📥 请求 {request.request_id} 加入等待队列 "
              f"(prompt={request.prompt_tokens}, max_new={request.max_new_tokens})")
    
    def schedule(self) -> dict:
        """
        调度下一步要执行的请求
        
        返回:
            scheduled_tokens: dict[request_id -> num_tokens]
        """
        scheduled_tokens: dict[str, int] = {}
        token_budget = self.config.max_num_batched_tokens
        
        # ========== Step 1: 调度已在运行的请求 (Decode) ==========
        for req in self.running_batch[:]:
            if req.is_prefilling:
                # Chunked Prefill: 继续 prefill
                remaining_prefill = req.prompt_tokens - req.computed_tokens
                if self.config.enable_chunked_prefill:
                    tokens_to_schedule = min(remaining_prefill, 
                                            self.config.chunk_size,
                                            token_budget)
                else:
                    tokens_to_schedule = min(remaining_prefill, token_budget)
            else:
                # Decode: 每次 1 个 token
                tokens_to_schedule = 1
            
            if tokens_to_schedule > 0 and token_budget >= tokens_to_schedule:
                scheduled_tokens[req.request_id] = tokens_to_schedule
                token_budget -= tokens_to_schedule
        
        # ========== Step 2: 从等待队列调度新请求 (Prefill) ==========
        while (self.waiting_queue and 
               len(self.running_batch) < self.config.max_num_seqs and
               token_budget > 0):
            
            req = self.waiting_queue[0]
            
            # 计算需要调度的 prefill tokens
            if self.config.enable_chunked_prefill:
                tokens_to_schedule = min(req.prompt_tokens,
                                        self.config.chunk_size,
                                        token_budget)
            else:
                # 不启用 chunked prefill，必须一次性处理完整 prompt
                if req.prompt_tokens > token_budget:
                    break
                tokens_to_schedule = req.prompt_tokens
            
            if tokens_to_schedule > 0:
                self.waiting_queue.popleft()
                req.status = "prefilling"
                self.running_batch.append(req)
                scheduled_tokens[req.request_id] = tokens_to_schedule
                token_budget -= tokens_to_schedule
        
        return scheduled_tokens
    
    def update(self, scheduled_tokens: dict[str, int]):
        """
        更新请求状态 (模拟执行完成)
        """
        current_time = time.time()
        finished_this_step = []
        
        for req in self.running_batch:
            if req.request_id not in scheduled_tokens:
                continue
            
            num_tokens = scheduled_tokens[req.request_id]
            req.computed_tokens += num_tokens
            
            # 检查是否完成 prefill
            if req.is_prefilling:
                if req.computed_tokens >= req.prompt_tokens:
                    req.status = "decoding"
            else:
                # Decode 阶段: 生成新 token
                req.generated_tokens += 1
                
                # 记录首 token 时间
                if req.first_token_time is None:
                    req.first_token_time = current_time
                
                # 检查是否完成
                if req.generated_tokens >= req.max_new_tokens:
                    req.status = "finished"
                    req.finish_time = current_time
                    finished_this_step.append(req)
        
        # 移除已完成的请求
        for req in finished_this_step:
            self.running_batch.remove(req)
            self.finished_requests.append(req)
        
        self.step_count += 1
        self.total_tokens_processed += sum(scheduled_tokens.values())
        
        return finished_this_step
    
    def get_batch_info(self) -> dict:
        """获取当前 batch 信息"""
        prefilling = [r for r in self.running_batch if r.is_prefilling]
        decoding = [r for r in self.running_batch if not r.is_prefilling]
        
        return {
            "step": self.step_count,
            "running": len(self.running_batch),
            "waiting": len(self.waiting_queue),
            "finished": len(self.finished_requests),
            "prefilling": len(prefilling),
            "decoding": len(decoding),
        }


# ============================================================================
# 2. 可视化输出
# ============================================================================

def print_batch_state(scheduler: ContinuousBatchingScheduler, 
                      scheduled_tokens: dict[str, int],
                      finished: list[Request]):
    """打印当前 batch 状态"""
    info = scheduler.get_batch_info()
    
    # 构建 batch 可视化
    batch_viz = []
    for req in scheduler.running_batch:
        tokens = scheduled_tokens.get(req.request_id, 0)
        if req.is_prefilling:
            progress = req.computed_tokens / req.prompt_tokens
            batch_viz.append(f"{req.request_id}[P{progress*100:.0f}%:{tokens}t]")
        else:
            progress = req.generated_tokens / req.max_new_tokens
            batch_viz.append(f"{req.request_id}[D{progress*100:.0f}%:{tokens}t]")
    
    # 打印状态
    print(f"\n{'─'*70}")
    print(f"Step {info['step']:3d} │ Running: {info['running']}/{scheduler.config.max_num_seqs} │ "
          f"Waiting: {info['waiting']} │ Finished: {info['finished']} │ "
          f"Prefill: {info['prefilling']} │ Decode: {info['decoding']}")
    print(f"        │ Batch: [{', '.join(batch_viz) if batch_viz else 'empty'}]")
    
    if finished:
        for req in finished:
            print(f"        │ ✅ {req.request_id} 完成! "
                  f"TTFT={req.ttft*1000:.1f}ms, Total={req.total_time*1000:.1f}ms, "
                  f"Tokens={req.generated_tokens}")


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# ============================================================================
# 3. 演示场景
# ============================================================================

def demo_continuous_batching():
    """
    演示 Continuous Batching 的工作流程
    """
    print_separator("Continuous Batching 演示")
    
    # 配置
    config = SchedulerConfig(
        max_num_seqs=4,              # 最多同时处理 4 个请求
        max_num_batched_tokens=1024, # 每步最多 1024 tokens
        enable_chunked_prefill=True,
        chunk_size=256,              # Prefill 分块大小
    )
    
    scheduler = ContinuousBatchingScheduler(config)
    
    print(f"\n调度器配置:")
    print(f"  - 最大并发请求: {config.max_num_seqs}")
    print(f"  - 每步最大 tokens: {config.max_num_batched_tokens}")
    print(f"  - Chunked Prefill: {'启用' if config.enable_chunked_prefill else '禁用'}")
    print(f"  - Chunk 大小: {config.chunk_size}")
    
    # ========== 场景 1: 初始请求 ==========
    print_separator("场景 1: 初始请求到达")
    
    initial_requests = [
        Request("A", prompt_tokens=512, max_new_tokens=100),
        Request("B", prompt_tokens=256, max_new_tokens=50),
        Request("C", prompt_tokens=128, max_new_tokens=30),
    ]
    
    for req in initial_requests:
        scheduler.add_request(req)
    
    # 运行几步
    for _ in range(8):
        scheduled = scheduler.schedule()
        finished = scheduler.update(scheduled)
        print_batch_state(scheduler, scheduled, finished)
        time.sleep(0.05)  # 模拟计算时间
    
    # ========== 场景 2: 新请求动态加入 ==========
    print_separator("场景 2: 新请求动态加入")
    print("  (模拟真实场景: 请求 B 完成后，新请求 D 立即加入)")
    
    new_request = Request("D", prompt_tokens=200, max_new_tokens=40)
    scheduler.add_request(new_request)
    
    for _ in range(10):
        # 随机添加新请求 (模拟真实流量)
        if random.random() < 0.2 and len(scheduler.waiting_queue) < 3:
            req_id = chr(ord('E') + len(scheduler.finished_requests) + 
                        len(scheduler.running_batch) + len(scheduler.waiting_queue) - 3)
            new_req = Request(
                req_id, 
                prompt_tokens=random.randint(64, 256),
                max_new_tokens=random.randint(20, 60)
            )
            scheduler.add_request(new_req)
        
        scheduled = scheduler.schedule()
        if not scheduled:
            break
        finished = scheduler.update(scheduled)
        print_batch_state(scheduler, scheduled, finished)
        time.sleep(0.05)
    
    # ========== 场景 3: 处理完所有请求 ==========
    print_separator("场景 3: 处理剩余请求")
    
    while scheduler.running_batch or scheduler.waiting_queue:
        scheduled = scheduler.schedule()
        if not scheduled:
            break
        finished = scheduler.update(scheduled)
        print_batch_state(scheduler, scheduled, finished)
        time.sleep(0.02)
    
    # ========== 统计信息 ==========
    print_separator("统计信息")
    
    print(f"\n  总步数: {scheduler.step_count}")
    print(f"  总处理 tokens: {scheduler.total_tokens_processed}")
    print(f"  完成请求数: {len(scheduler.finished_requests)}")
    
    if scheduler.finished_requests:
        ttfts = [r.ttft for r in scheduler.finished_requests if r.ttft]
        total_times = [r.total_time for r in scheduler.finished_requests if r.total_time]
        
        print(f"\n  平均 TTFT: {sum(ttfts)/len(ttfts)*1000:.1f} ms")
        print(f"  平均总时间: {sum(total_times)/len(total_times)*1000:.1f} ms")
        
        print(f"\n  各请求详情:")
        for req in scheduler.finished_requests:
            print(f"    {req.request_id}: prompt={req.prompt_tokens}, "
                  f"generated={req.generated_tokens}, "
                  f"TTFT={req.ttft*1000:.1f}ms, "
                  f"Total={req.total_time*1000:.1f}ms")


def demo_comparison():
    """
    对比 Static Batching vs Continuous Batching
    """
    print_separator("Static vs Continuous Batching 对比")
    
    requests = [
        ("A", 100, 200),  # 长请求
        ("B", 100, 50),   # 短请求
        ("C", 100, 80),   # 中等请求
    ]
    
    # ========== Static Batching 模拟 ==========
    print("\n┌─────────────── Static Batching ───────────────┐")
    print("│ 所有请求必须等待最长的请求完成                   │")
    
    max_tokens = max(t[2] for t in requests)
    total_steps = 100 + max_tokens  # prefill + max decode
    
    print(f"│                                               │")
    print(f"│ 总步数: {total_steps} (受限于最长请求 A)          │")
    print(f"│                                               │")
    print(f"│ 时间线:                                       │")
    print(f"│ Step 1-100:   [A, B, C] prefill               │")
    print(f"│ Step 101-150: [A, -, C] B完成,GPU空闲50%        │")
    print(f"│ Step 151-180: [A, -, -] C完成,GPU空闲67%        │")
    print(f"│ Step 181-300: [A, -, -] 只有A在运行,GPU空闲67%  │")
    print(f"│                                               │")
    print(f"│ GPU 利用率: ~45%                              │")
    print("└───────────────────────────────────────────────┘")
    
    # ========== Continuous Batching 模拟 ==========
    print("\n┌─────────────── Continuous Batching ────────────┐")
    print("│ 完成的请求立即释放,新请求随时加入               │")
    print(f"│                                               │")
    print(f"│ 时间线:                                       │")
    print(f"│ Step 1-100:   [A, B, C] prefill               │")
    print(f"│ Step 101-150: [A, D, C] B完成,D立即加入        │")
    print(f"│ Step 151-180: [A, D, E] C完成,E立即加入        │")
    print(f"│ Step 181-200: [A, F, E] D完成,F立即加入        │")
    print(f"│ ...          [持续处理新请求]                 │")
    print(f"│                                               │")
    print(f"│ GPU 利用率: ~95%+                             │")
    print("└───────────────────────────────────────────────┘")
    
    # 数值对比
    print("\n性能对比:")
    print("┌─────────────────┬───────────────┬───────────────────┐")
    print("│ 指标            │ Static        │ Continuous        │")
    print("├─────────────────┼───────────────┼───────────────────┤")
    print("│ GPU 利用率      │ ~45%          │ ~95%+             │")
    print("│ 请求 B 等待时间 │ 300 steps     │ 150 steps         │")
    print("│ 吞吐量          │ 1x            │ 2-3x              │")
    print("│ 内存效率        │ 低 (预分配)   │ 高 (动态分配)     │")
    print("└─────────────────┴───────────────┴───────────────────┘")


# ============================================================================
# 4. 运行演示
# ============================================================================

if __name__ == "__main__":
    demo_continuous_batching()
    print("\n")
    demo_comparison()
    
    print("\n" + "="*70)
    print("  演示完成!")
    print("="*70)
