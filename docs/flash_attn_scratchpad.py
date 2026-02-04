import torch
import torch.nn.functional as F
import math

def manual_attention_standard(q, k, v):
    """
    Standard Attention Implementation (PyTorch Reference)
    标准注意力实现（PyTorch 参考版本）
    
    O(N^2) memory usage effectively (stores full score matrix)
    有效内存使用为 O(N^2)（需要存储完整的注意力分数矩阵）
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    # (B, N, d) @ (B, d, N) -> (B, N, N)
    # 矩阵乘法：(批次, 序列长度, 维度) @ (批次, 维度, 序列长度) -> (批次, 序列长度, 序列长度)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    # (B, N, N) @ (B, N, d) -> (B, N, d)
    # 矩阵乘法：(批次, 序列长度, 序列长度) @ (批次, 序列长度, 维度) -> (批次, 序列长度, 维度)
    output = torch.matmul(attn, v)
    return output

def manual_attention_tiled(q, k, v, block_size=2):
    """
    Simulating FlashAttention Tiling Logic (Python Version)
    模拟 FlashAttention 的分块计算逻辑（Python 版本）
    
    Key Concepts from Lecture 06 & FlashAttention Paper:
    来自 Lecture 06 和 FlashAttention 论文的核心概念：
    
    1. Tiling: Load blocks of Q, K, V into SRAM (simulated by processing small chunks)
       分块：将 Q, K, V 的块加载到 SRAM（通过处理小块来模拟）
    
    2. Online Softmax: Compute Softmax one block at a time without seeing the full row first.
       在线 Softmax：无需先看到整行数据，即可逐块计算 Softmax
    
    Variables mapping to paper:
    变量与论文的对应关系：
    - O: Output matrix / 输出矩阵
    - L: Denominator (sum of exps) / 分母（指数和）
    - M: Max value (for numerical stability) / 最大值（用于数值稳定性）
    """
    B, N, d = q.shape
    scale = 1.0 / math.sqrt(d)
    
    # Initialize buffers (HBM simulation)
    # 初始化缓冲区（模拟 HBM 高带宽内存）
    # O: Final result / 最终结果
    # L: Interaction terms (sum of exps), corresponds to 'l' in paper
    #    交互项（指数和），对应论文中的 'l'
    # M: Running maximum, corresponds to 'm' in paper
    #    运行中的最大值，对应论文中的 'm'
    O = torch.zeros_like(q)
    L = torch.zeros(B, N, 1)
    M = torch.full((B, N, 1), float('-inf'))

    # Outer Loop: Iterate over Q blocks (Rows)
    # 外层循环：遍历 Q 的块（行方向）
    # Corresponds to: for i = 1 to Tr in paper
    # 对应论文中：for i = 1 to Tr
    for i in range(0, N, block_size):
        # Load Q block from HBM to SRAM
        # 将 Q 块从 HBM 加载到 SRAM
        q_i = q[:, i:i+block_size, :]  # Shape: (B, Br, d) / 形状：(批次, 块行数, 维度)
        
        # Inner Loop: Iterate over K, V blocks (Columns)
        # 内层循环：遍历 K, V 的块（列方向）
        # Corresponds to: for j = 1 to Tc in paper
        # 对应论文中：for j = 1 to Tc
        for j in range(0, N, block_size):
            # Load K, V blocks from HBM to SRAM
            # 将 K, V 块从 HBM 加载到 SRAM
            k_j = k[:, j:j+block_size, :]  # Shape: (B, Bc, d) / 形状：(批次, 块列数, 维度)
            v_j = v[:, j:j+block_size, :]  # Shape: (B, Bc, d) / 形状：(批次, 块列数, 维度)
            
            # --- Algorithm Core ---
            # --- 算法核心 ---
            
            # 1. Compute Score for this block: S_ij = Q_i * K_j^T
            #    计算当前块的注意力分数：S_ij = Q_i * K_j^T
            s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale # (B, Br, Bc) / (批次, 块行数, 块列数)
            
            # 2. Local max for this block
            #    当前块的局部最大值
            m_ij, _ = torch.max(s_ij, dim=-1, keepdim=True) # (B, Br, 1) / (批次, 块行数, 1)
            
            # 3. Update Global max M_new
            #    更新全局最大值 M_new
            # M_new = max(M_prev, m_ij)
            m_prev = M[:, i:i+block_size, :]
            m_new = torch.maximum(m_prev, m_ij)
            
            # 4. Compute exp(S_ij - M_new) -> P_ij
            #    计算 exp(S_ij - M_new) -> P_ij
            # Using new max for stability
            # 使用新的最大值以保证数值稳定性
            p_ij = torch.exp(s_ij - m_new)
            
            # 5. Update L (Denominator)
            #    更新 L（分母）
            # We need to rescale the previous L because the max M changed!
            # 因为最大值 M 发生了变化，我们需要重新缩放之前的 L！
            # Rescale factor alpha = exp(M_prev - M_new)
            # 缩放因子 alpha = exp(M_prev - M_new)
            # If M_new == M_prev (current block isn't larger), alpha = 1
            # 如果 M_new == M_prev（当前块不是更大的），alpha = 1
            # If M_new > M_prev, alpha < 1 (shrinks old values)
            # 如果 M_new > M_prev，alpha < 1（缩小旧值）
            alpha = torch.exp(m_prev - m_new)
            l_prev = L[:, i:i+block_size, :]
            
            # Sum of exps for current block
            # 当前块的指数和
            l_block = torch.sum(p_ij, dim=-1, keepdim=True)
            
            # New global sum
            # 新的全局累加和
            l_new = alpha * l_prev + l_block
            
            # 6. Update Output O
            #    更新输出 O
            # Rescale previous output O_prev by alpha
            # 使用 alpha 缩放之前的输出 O_prev
            # Add current block's contribution: P_ij * V_j
            # 添加当前块的贡献：P_ij * V_j
            o_prev = O[:, i:i+block_size, :]
            pv_block = torch.matmul(p_ij, v_j)
            
            o_new = alpha * o_prev + pv_block
            
            # --- Write back to HBM ---
            # --- 写回 HBM ---
            M[:, i:i+block_size, :] = m_new
            L[:, i:i+block_size, :] = l_new
            O[:, i:i+block_size, :] = o_new

    # 7. Final Normalization
    #    最终归一化
    # O contains the weighted sum of Vs. We need to divide by the total weight (L).
    # O 包含 V 的加权和。我们需要除以总权重 (L)。
    O = O / L
    
    return O

# --- Test Bench ---
# --- 测试代码 ---
if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, d = 1, 8, 4  # Small scale for debugging / 小规模用于调试
    
    Q = torch.randn(B, N, d)
    K = torch.randn(B, N, d)
    V = torch.randn(B, N, d)
    
    print(f"Shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"形状：Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    print("Running Standard Attention...")
    print("运行标准注意力...")
    expected = manual_attention_standard(Q, K, V)
    
    print("Running Tiled Attention (Flash Logic)...")
    print("运行分块注意力（Flash 逻辑）...")
    # block_size=4 means 2 steps for N=8
    # block_size=4 意味着对于 N=8 需要 2 步
    actual = manual_attention_tiled(Q, K, V, block_size=4) 
    
    # Check correctness
    # 检查正确性
    diff = torch.abs(expected - actual).max()
    print(f"Max Difference: {diff.item()}")
    print(f"最大差异：{diff.item()}")
    
    if diff < 1e-5:
        print("✅ Success! Tiled implementation matches Standard Attention.")
        print("✅ 成功！分块实现与标准注意力匹配。")
    else:
        print("❌ Mismatch!")
        print("❌ 不匹配！")
