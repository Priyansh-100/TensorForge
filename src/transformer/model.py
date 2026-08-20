"""
Transformer from scratch — built component by component.

Architecture (Vaswani et al. 2017):
                    Output
                      ↑
                 Linear + Softmax
                      ↑
                Add & Norm
                  ↑   ↑
              Feed Forward
                  ↑   ↑
                Add & Norm
                  ↑   ↑
          Multi-Head Attention  ←── K, V from Encoder
                  ↑   ↑
              Positional Encoding
                  ↑   ↑
             Embedding (target)
                      ↑
                 <shifted right>

                    ↑
                Add & Norm
                  ↑   ↑
              Feed Forward
                  ↑   ↑
                Add & Norm
                  ↑   ↑
          Multi-Head Attention ←── Q = K = V
                  ↑   ↑
              Positional Encoding
                  ↑   ↑
             Embedding (source)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,  # (batch, heads, seq_len, d_k)
    K: torch.Tensor,  # (batch, heads, seq_len, d_k)
    V: torch.Tensor,  # (batch, heads, seq_len, d_k)
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    mask: True = allowed to attend, False = masked out
    Returns (output, attention_weights).
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE) — how Llama/GPT-neoX encode position
# ---------------------------------------------------------------------------
#
# Instead of adding a position vector to the embedding, RoPE *rotates* the
# query and key vectors. For each pair of dims (2i, 2i+1) with frequency w_i:
#     x'_{2i}   = x_{2i} cos(pos·w_i) - x_{2i+1} sin(pos·w_i)
#     x'_{2i+1} = x_{2i} sin(pos·w_i) + x_{2i+1} cos(pos·w_i)
#
# The magic: the dot product between a query at position m and key at
# position n becomes q^T R(m-n) k — a function of the *relative* offset
# m-n only. That relative-position bias transfers to longer sequences
# than were seen in training (great for length extrapolation).

def precompute_rope(dim: int, max_len: int, base: float = 10000.0):
    """Returns (cos, sin) tensors of shape (max_len, dim), one angle per pair."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(max_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)            # (max_len, dim // 2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (max_len, dim); angles per pair
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split in half and swap with negation: pairs dim i with dim i + d/2.
    (This is the convention compatible with cat([freqs, freqs]) below.)"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., dim), cos/sin: broadcastable to (..., dim). Rotates in place of adding."""
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# 2. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        num_kv_heads: int | None = None,
    ):
        """
        num_kv_heads: heads that get their OWN K/V (Grouped Query Attention).
        Each query head belongs to a group of num_heads // num_kv_heads heads
        sharing ONE key/value head — the cache becomes num_kv_heads/num_heads
        smaller. num_kv_heads == num_heads is plain multi-head attention
        (every head has its own K/V); this is what LLaMA-2/3 & Mistral use.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be a multiple of "
                             f"num_kv_heads ({num_kv_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads  # query heads sharing one KV head
        self.d_k = d_model // num_heads
        self.d_kv = self.d_k  # KV heads share the query head width; GQA shrinks the COUNT

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_kv)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_kv)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights: torch.Tensor | None = None  # for visualization

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        rope_cos_k: torch.Tensor | None = None,
        rope_sin_k: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        mask:    bool, True = allowed to attend.
        kv_cache: (cached_K, cached_V) from previous steps. When provided,
                  K/V for the CURRENT tokens are appended and the new cache
                  is returned: generational K/V for past tokens don't change,
                  so we never recompute them.
        rope_cos/rope_sin: (seq, d_model) rotation tables for Q positions.
                  rope_cos_k/rope_sin_k: tables for K positions — K lives in a
                  num_kv_heads·d_kv-dimensional space, so under GQA its table
                  has a different width. Falls back to the Q tables (full MHA).
        Returns: output always; (output, new_cache) when kv_cache is given.
        """
        batch_size = Q.size(0)

        # In cross-attention, K/V (from encoder, batch=1) can differ from Q (beams)
        if K.size(0) != batch_size:
            K = K.expand(batch_size, -1, -1).contiguous()
            V = V.expand(batch_size, -1, -1).contiguous()

        # 1. Linear projections. Q keeps one head per head; K/V get fewer columns.
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)  # (batch, seq_len, num_kv_heads * d_kv)
        V = self.W_v(V)

        # RoPE: rotate in projected space (K space is smaller under GQA, so its
        # rotation table differs from Q's — the caller supplies both).
        if rope_cos is not None and rope_sin is not None:
            Q = apply_rope(Q, rope_cos.unsqueeze(0), rope_sin.unsqueeze(0))
            k_cos = rope_cos_k if rope_cos_k is not None else rope_cos
            k_sin = rope_sin_k if rope_sin_k is not None else rope_sin
            K = apply_rope(K, k_cos.unsqueeze(0), k_sin.unsqueeze(0))

        # 2. Reshape for multi-head: (batch, seq_len, dim) -> (batch, heads, seq_len, d)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_kv_heads, self.d_kv).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_kv_heads, self.d_kv).transpose(1, 2)

        # 2.5 KV cache: append in the SMALL (num_kv_heads) representation —
        # caching the expanded form would throw the cache saving away.
        if kv_cache is not None:
            cached_K, cached_V = kv_cache
            K = torch.cat([cached_K, K], dim=2)
            V = torch.cat([cached_V, V], dim=2)
            new_cache = (K.detach(), V.detach())

        if self.num_kv_heads != self.num_heads:
            K = K.repeat_interleave(self.group_size, dim=1)  # (B, num_heads, T, d_k)
            V = V.repeat_interleave(self.group_size, dim=1)

        # 3. Scaled dot-product attention per head
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        self.last_attn_weights = attn_weights.detach()  # (batch, heads, q_len, k_len)

        # 4. Concatenate heads: (batch, heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. Final linear projection
        output = self.W_o(attn_output)

        if kv_cache is not None:
            return output, new_cache
        return output


# ---------------------------------------------------------------------------
# Learning rate scheduler (from the original paper)
# ---------------------------------------------------------------------------

class NoamSchedule(torch.optim.lr_scheduler.LambdaLR):
    """
    lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Phase 1 (warmup):    lr grows linearly from 0 -> peak at step = warmup
    Phase 2 (decay):     lr shrinks as 1/sqrt(step)
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        def lr(step: int) -> float:
            return self.d_model**-0.5 * min(
                (step + 1) ** -0.5, (step + 1) * self.warmup_steps**-1.5
            )

        super().__init__(optimizer, lr)


# ---------------------------------------------------------------------------
# 3. Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 4. Position-Wise Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ---------------------------------------------------------------------------
# 5. Encoder Layer
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 6. Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        look_ahead_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention (prevents looking ahead)
        attn_out = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention: Q from decoder, K/V from encoder
        attn_out = self.cross_attn(x, encoder_output, encoder_output, pad_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ---------------------------------------------------------------------------
# 7. Encoder (stack of EncoderLayers)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ---------------------------------------------------------------------------
# 8. Decoder (stack of DecoderLayers)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        look_ahead_mask: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, look_ahead_mask, pad_mask)
        return x


# ---------------------------------------------------------------------------
# 9. Transformer (full model)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return self.output_proj(decoder_output)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 10. Mask helpers
# ---------------------------------------------------------------------------

def create_pad_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """Mask padding tokens. Shape: (batch, 1, 1, seq_len). True = keep."""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(sz: int) -> torch.Tensor:
    """Lower-triangular mask — True = allowed to attend. Shape: (1, 1, sz, sz)."""
    mask = torch.tril(torch.ones(sz, sz, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def create_masks(
    src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    src_mask = create_pad_mask(src, pad_idx)
    tgt_pad_mask = create_pad_mask(tgt, pad_idx)

    look_ahead = create_look_ahead_mask(tgt.size(1))
    tgt_mask = tgt_pad_mask & look_ahead.to(tgt.device)
    return src_mask, tgt_mask
