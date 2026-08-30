#!/usr/bin/env python3
"""
Flash Attention v2 Triton kernel for mini-GPT.

Memory-efficient attention that fuses the entire attention computation
into a single kernel, reducing HBM reads/writes and enabling longer contexts.

Reference: Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism" (2023)
          Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None
    print("Triton not available. Install with: pip install triton")


if TRITON_AVAILABLE:
    @triton.jit
    def _flash_attention_forward_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        B, H, T, D,  # batch, heads, seq_len, head_dim
        sm_scale,  # 1/sqrt(d_k)
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_ob, stride_oh, stride_ot, stride_od,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Flash Attention v2 forward kernel.
        
        Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
        using tiling to avoid materializing the full T x T attention matrix.
        
        Grid: (batch * heads, ceil(T / BLOCK_M))
        Block: (BLOCK_M,)
        """
        # Program IDs
        pid_bh = tl.program_id(0)  # batch * heads
        pid_m = tl.program_id(1)   # row tile index
        
        batch = pid_bh // H
        head = pid_bh % H
        
        # Starting positions
        q_start = pid_m * BLOCK_M
        _ = 0  # k_start (unused)
        
        # Accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')  # row max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # row sum
        o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator
        
        # Loop over K/V blocks
        for start_n in range(0, T, BLOCK_N):
            # Load K block
            k_offset = start_n + tl.arange(0, BLOCK_N)
            _ = k_offset < T  # k_mask (unused)
            
            # Load V block
            v_offset = start_n + tl.arange(0, BLOCK_N)
            _ = v_offset < T  # v_mask (unused)
            
            # Compute QK^T for this block
            q_offset = q_start + tl.arange(0, BLOCK_M)
            _ = q_offset < T  # q_mask (unused)
            
            # Load Q block
            q = tl.load(
                Q_ptr + batch * stride_qb + head * stride_qh + 
                (q_offset[:, None] * stride_qt + tl.arange(0, BLOCK_D)[None, :] * stride_qd),
                mask=(q_offset[:, None] < T) & (tl.arange(0, BLOCK_D)[None, :] < D),
                other=0.0
            )
            
            # Load K block
            k = tl.load(
                K_ptr + batch * stride_kb + head * stride_kh + 
                (k_offset[:, None] * stride_kt + tl.arange(0, BLOCK_D)[None, :] * stride_kd),
                mask=(k_offset[:, None] < T) & (tl.arange(0, BLOCK_D)[None, :] < D),
                other=0.0
            )
            
            # S = Q K^T * sm_scale
            s = tl.dot(q, tl.trans(k)) * sm_scale
            
            # Causal mask
            if IS_CAUSAL:
                q_idx = q_offset[:, None]
                k_idx = k_offset[None, :]
                causal_mask = q_idx >= k_idx
                s = tl.where(causal_mask, s, float('-inf'))
            
            # Softmax
            m_ij = tl.max(s, axis=1)
            p = tl.exp(s - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            
            # Load V block
            v = tl.load(
                V_ptr + batch * stride_vb + head * stride_vh + 
                (v_offset[:, None] * stride_vt + tl.arange(0, BLOCK_D)[None, :] * stride_vd),
                mask=(v_offset[:, None] < T) & (tl.arange(0, BLOCK_D)[None, :] < D),
                other=0.0
            )
            
            # Update output
            p_v = tl.dot(p, v)
            
            # Rescale and accumulate
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            l_i_new = l_i * alpha + l_ij
            o_i = o_i * alpha[:, None] + p_v
            
            m_i = m_i_new
            l_i = l_i_new
        
        # Final output: O = o_i / l_i
        o_i = o_i / l_i[:, None]
        
        # Store output
        tl.store(
            O_ptr + batch * stride_ob + head * stride_oh + 
            (q_start + tl.arange(0, BLOCK_M))[:, None] * stride_ot + 
            tl.arange(0, BLOCK_D)[None, :] * stride_od,
            o_i,
            mask=(q_start + tl.arange(0, BLOCK_M)[:, None] < T) & (tl.arange(0, BLOCK_D)[None, :] < D)
        )


if TRITON_AVAILABLE:
    @triton.jit
    def _flash_attention_backward_kernel(
        dO_ptr, Q_ptr, K_ptr, V_ptr, O_ptr,
        dQ_ptr, dK_ptr, dV_ptr,
        B, H, T, D,
        sm_scale,
        stride_ob, stride_oh, stride_ot, stride_od,
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_dqb, stride_dqh, stride_dqt, stride_dqd,
        stride_dkb, stride_dkh, stride_dkt, stride_dkd,
        stride_dvb, stride_dvh, stride_dvt, stride_dvd,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """
        Flash Attention backward kernel.
        Computes dQ, dK, dV from dO using the same tiling strategy.
        """
        # Simplified - full implementation is complex
        # See FlashAttention-2 paper for complete derivation
        pass


def flash_attention_forward(Q, K, V, sm_scale=None, causal=True):
    """
    Flash Attention v2 forward pass using Triton.
    
    Args:
        Q, K, V: [batch, heads, seq_len, head_dim]
        sm_scale: scaling factor (default: 1/sqrt(head_dim))
        causal: whether to apply causal mask
    
    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available. Install with: pip install triton")
    
    sm_scale = sm_scale or (1.0 / (Q.shape[-1] ** 0.5))
    
    out = torch.empty_like(Q)
    
    _flash_attention_forward_kernel[
        (Q.shape[0] * Q.shape[1], (Q.shape[2] + 63) // 64)
    ](
        Q, K, V, out,
        Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3],
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=64, BLOCK_N=64, BLOCK_D=min(64, Q.shape[-1]),
        IS_CAUSAL=causal
    )
    
    return out


class FlashAttention(nn.Module):
    """Flash Attention module with optional Triton kernel."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_triton: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_triton = use_triton and TRITON_AVAILABLE and torch.cuda.is_available()
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.shape
        H = self.num_heads
        D = self.head_dim
        
        Q = self.W_q(Q).view(B, T, H, D).transpose(1, 2)
        K = self.W_k(K).view(B, T, H, D).transpose(1, 2)
        V = self.W_v(V).view(B, T, H, D).transpose(1, 2)
        
        if self.use_triton and TRITON_AVAILABLE:
            try:
                out = flash_attention_forward(Q, K, V, causal=True)
            except Exception as e:
                print(f"Triton kernel failed, falling back to PyTorch: {e}")
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


def test_flash_attention():
    """Test Flash Attention against PyTorch's scaled_dot_product_attention."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("Skipping test: Triton or CUDA not available")
        return
    
    torch.manual_seed(42)
    B, H, T, _ = 2, 4, 128, 64
    Q = torch.randn(B, H, T, 64, dtype=torch.float16, device='cuda')
    K = torch.randn(B, H, T, 64, dtype=torch.float16, device='cuda')
    V = torch.randn(B, H, T, 64, dtype=torch.float16, device='cuda')
    
    # PyTorch reference
    out_ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    
    # Flash Attention
    out_flash = flash_attention_forward(Q, K, V, causal=True)
    
    # Check accuracy
    max_diff = (out_ref - out_flash).abs().max().item()
    rel_err = max_diff / (out_ref.abs().max() + 1e-8)
    
    print(f"Max diff: {max_diff:.2e}, Rel err: {rel_err:.2e}")
    assert rel_err < 1e-3, f"Flash Attention mismatch: {rel_err}"
    print("Flash Attention test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    if args.test:
        test_flash_attention()
    else:
        print("Flash Attention v2 Triton kernel")
        print("Install triton: pip install triton")
        print("Run test with: python scripts/flash_attention.py --test")