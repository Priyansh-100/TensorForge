"""
Mini-GPT library: a decoder-only transformer for character-level modeling.

The insight: for language modeling you don't need an encoder — you only need
the *masked* self-attention part (each token attends to past tokens only).
That's exactly what the decoder half of a seq2seq transformer does, minus
the cross-attention.

CLI entry point: scripts/gpt.py
"""

import math
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer.model import (
    MultiHeadAttention,
    FeedForward,
    NoamSchedule,
    create_look_ahead_mask,
    precompute_rope,
)

# ---------------------------------------------------------------------------
# Data: character-level tokenizer
# ---------------------------------------------------------------------------


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CharDataset(Dataset):
    """Slices the whole corpus into (input, target) pairs at random offsets."""

    def __init__(self, data: torch.Tensor, block_size: int, num_pairs: int):
        self.data = data
        self.block_size = block_size
        self.num_pairs = num_pairs

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, _):
        # Random start position so every epoch sees different slices.
        # A corpus shorter than block_size+2 would collapse randint's range.
        hi = len(self.data) - self.block_size - 1
        idx = torch.randint(0, max(hi, 1), ())
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Temperature + optional nucleus (top-p) sampling. top_p=1.0 is plain
    temperature sampling; < 1.0 restricts the draw to the smallest set of
    tokens whose cumulative probability exceeds top_p (Holtzman et al. 2019)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
        keep[..., 0] = True  # always keep the most likely token
        trimmed = sorted_probs.masked_fill(~keep, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx,
                                                 trimmed / trimmed.sum(dim=-1, keepdim=True))
    return torch.multinomial(probs, num_samples=1)


class GPTBlock(nn.Module):
    """Decoder-only block: masked self-attention + FFN (no cross-attention)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 num_kv_heads: int | None = None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,
                                            num_kv_heads=num_kv_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cos_k: torch.Tensor | None = None,
        sin_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask, rope_cos=cos,
                                                       rope_sin=sin, rope_cos_k=cos_k,
                                                       rope_sin_k=sin_k)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

    def step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        cos_k: torch.Tensor | None = None,
        sin_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental generation: reuse cached K/V from all past tokens."""
        if cache is None:
            attn = self.self_attn
            B = x.size(0)
            empty = torch.zeros(B, attn.num_kv_heads, 0, attn.d_kv, device=x.device, dtype=x.dtype)
            cache = (empty, empty)
        attn_out, cache = self.self_attn(x, x, x, mask=mask, kv_cache=cache, rope_cos=cos,
                                         rope_sin=sin, rope_cos_k=cos_k, rope_sin_k=sin_k)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, cache


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
        rope: bool = False,
        num_kv_heads: int | None = None,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        if rope:
            # Positions live inside attention (rotated Q/K) — no embedding needed
            cos, sin = precompute_rope(d_model, max_len)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
            # Under GQA, K lives in a num_kv_heads·d_k-wide space → own table.
            # persistent=False: derived state, excluded from checkpoints.
            num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
            kv_dim = num_kv_heads * (d_model // num_heads)
            cos_k, sin_k = precompute_rope(kv_dim, max_len)
            self.register_buffer("cos_k", cos_k, persistent=False)
            self.register_buffer("sin_k", sin_k, persistent=False)
        else:
            self.pos_embedding = nn.Embedding(max_len, d_model)  # learned positions
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, num_heads, d_ff, dropout, num_kv_heads=num_kv_heads)
             for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.tie_weights = tie_weights
        if tie_weights:
            # GPT-2 style weight tying: the embedding and the output head are the
            # SAME matrix — the model only learns one set of token vectors, used
            # both as input embedding and output logits. Saves vocab·d_model
            # parameters; usually a small quality bump.
            self.lm_head.weight = self.token_embedding.weight

        self.max_len = max_len
        self.num_heads = num_heads
        self.num_kv_heads = self.blocks[0].self_attn.num_kv_heads
        self.rope = rope

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _grow_rope_tables(self, need: int):
        """RoPE frequencies are deterministic, so the cos/sin tables can be
        extended on demand — positions past max_len aren't a limit, just a
        lazily-fixed cache size. (Learned positions have no such escape.)"""
        if self.rope and need > self.cos.size(0):
            size = max(need, self.cos.size(0) * 2)  # geometric growth, not 1 token at a time
            cos, sin = precompute_rope(self.cos.size(1), size)
            self.register_buffer("cos", cos.to(self.cos.device))
            self.register_buffer("sin", sin.to(self.sin.device))
            cos_k, sin_k = precompute_rope(self.cos_k.size(1), size)
            self.register_buffer("cos_k", cos_k.to(self.cos_k.device), persistent=False)
            self.register_buffer("sin_k", sin_k.to(self.sin_k.device), persistent=False)

    def forward(
        self, idx: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_embedding(idx)  # (B, T, d_model)

        if self.rope:
            self._grow_rope_tables(T)
            cos = self.cos[:T].unsqueeze(0)  # (T, d_model) → broadcast over batch
            sin = self.sin[:T].unsqueeze(0)
            cos_k = self.cos_k[:T].unsqueeze(0)  # (T, num_kv_heads·d_k)
            sin_k = self.sin_k[:T].unsqueeze(0)
            pos = None
        else:
            cos = sin = cos_k = sin_k = None
            pos = self.pos_embedding(torch.arange(T, device=idx.device))

        x = x + pos if pos is not None else x

        for block in self.blocks:
            x = block(x, mask, cos=cos, sin=sin, cos_k=cos_k, sin_k=sin_k)
        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab)

    def _cached_forward(
        self,
        idx: torch.Tensor,                          # (B, T')
        caches: list,                               # one (K, V) cache per block
        start_pos: int,                             # absolute position of idx[0]
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Run blocks over idx tokens, threading KV caches through. Positions
        matter for RoPE (rotations depend on absolute position), so cos/sin are
        sliced at (start_pos .. start_pos+T)."""
        B, T = idx.shape
        x = self.token_embedding(idx)

        if self.rope:
            self._grow_rope_tables(start_pos + T)
            cos = self.cos[start_pos : start_pos + T]
            sin = self.sin[start_pos : start_pos + T]
            cos_k = self.cos_k[start_pos : start_pos + T]
            sin_k = self.sin_k[start_pos : start_pos + T]
            pos = None
        else:
            pos = self.pos_embedding(
                torch.arange(start_pos, start_pos + T, device=idx.device)
            )
            cos = sin = cos_k = sin_k = None

        x = x + pos if pos is not None else x  # added ONCE before the blocks

        for i, block in enumerate(self.blocks):
            x, caches[i] = block.step(x, caches[i], mask=mask,
                                      cos=cos.unsqueeze(0) if cos is not None else None,
                                      sin=sin.unsqueeze(0) if sin is not None else None,
                                      cos_k=cos_k.unsqueeze(0) if cos_k is not None else None,
                                      sin_k=sin_k.unsqueeze(0) if sin_k is not None else None)
        return x, caches

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 1.0) -> torch.Tensor:
        """Autoregressive sampling; idx is (B, T) initial context."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len :]
            mask = create_look_ahead_mask(idx_cond.size(1)).to(idx.device)
            logits = self(idx_cond, mask)  # (B, T, vocab)
            next_idx = _sample_token(logits[:, -1, :], temperature, top_p)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def generate_cached(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                        top_p: float = 1.0) -> torch.Tensor:
        """
        KV-cache generation: each step only computes attention for the ONE new
        token (the K/V of every past token are reused). Naive generation calls
        forward() over the whole prefix each step → O(T²) work. Caching makes
        inference O(T) — this is what llama.cpp / vLLM / HF do.
        """
        caches = [None] * len(self.blocks)
        B, context_T = idx.shape

        # Learned position embeddings can't index beyond max_len (RoPE has no such cap)
        if not self.rope and context_T + max_new_tokens > self.max_len:
            raise ValueError(
                f"Learned positions only reach {self.max_len} (need {context_T + max_new_tokens}); "
                f"use --rope for arbitrary-length generation."
            )

        # Prefill: process the whole prompt WITH the causal mask (crucial — without
        # it, intermediate positions attend to future tokens and their hidden
        # states become wrong keys for every later layer). create_look_ahead_mask
        # already returns (1, 1, T, T).
        prefill_mask = create_look_ahead_mask(context_T).to(idx.device)
        with torch.no_grad():
            x, caches = self._cached_forward(idx, caches, 0, prefill_mask)

        for step in range(max_new_tokens):
            logits = self.lm_head(self.ln_f(x[:, -1:, :]))  # (B, 1, vocab)
            next_idx = _sample_token(logits[:, -1, :], temperature, top_p)  # (B, 1)

            idx = torch.cat([idx, next_idx], dim=1)

            # Single new token: all cached keys are <= its position → no mask needed
            # (a causal row for the last position is all-True anyway).
            with torch.no_grad():
                x, caches = self._cached_forward(next_idx, caches, context_T + step)

        return idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class _TBLogger:
    """Optional TensorBoard sink. Import is lazy so the core package never
    depends on the `tensorboard` pip package — logging just turns off with a
    note when it isn't installed."""

    def __init__(self, log_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            print(f"  tensorboard not installed — `pip install tensorboard` to log to {log_dir}")

    def add_scalars(self, tag: str, values: dict, step: int):
        if self.writer is not None:
            self.writer.add_scalars(tag, values, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def train(
    tokenizer: CharTokenizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    epochs: int,
    block_size: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    batch_size: int,
    save: bool,
    rope: bool = False,
    save_path: str = "gpt.pt",
    num_kv_heads: int | None = None,
    seed: int | None = None,
    amp: bool = False,
    grad_accum: int = 1,
    compile_model: bool = False,
    tb_log: str | None = None,
    num_pairs: int = 20000,
    num_val_pairs: int = 2000,
    tie_weights: bool = False,
) -> GPT:
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed: {seed} — model init, data sampling and shuffling are reproducible")

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=block_size,
        rope=rope,
        num_kv_heads=num_kv_heads,
        tie_weights=tie_weights,
    ).to(device)
    print(f"Model parameters: {model.count_params():,}")

    # torch.compile: traced/optimized kernels (Inductor). No API change.
    if compile_model:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("  torch.compile: ON")
        except Exception as e:  # e.g. unsupported backend on some MPS builds
            print(f"  torch.compile unavailable: {e}")

    # AMP: fp16 autocast + GradScaler. Only CUDA/MPS have a fp16 path here.
    use_amp = amp and device.type in ("cuda", "mps")
    if amp and not use_amp:
        print("  AMP: fp16 autocast needs CUDA or MPS — running in fp32 on CPU")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    logger = _TBLogger(tb_log) if tb_log else None

    train_ds = CharDataset(train_data, block_size, num_pairs=num_pairs)
    val_ds = CharDataset(val_data, block_size, num_pairs=num_val_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=1000)

    # No padding anywhere → mask is always the causal one
    mask = create_look_ahead_mask(block_size).to(device)

    toks_per_step = batch_size * block_size
    best_val = float("inf")
    step = 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext():
                logits = model(x, mask)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            scaler.scale(loss).backward() if scaler is not None else loss.backward()

            if i % grad_accum == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            total_loss += loss.item() * grad_accum

        # Validation (no dropout, no grad, fp32 for stable metrics)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, mask)
                val_loss += criterion(logits.view(-1, logits.size(-1)), y.view(-1)).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        ppl = math.exp(avg_val)
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | train {avg_train:.4f} | val {avg_val:.4f} | perplexity {ppl:.2f} | lr {lr_now:.2e}")

        if logger is not None:
            logger.add_scalars("train", {"loss": avg_train, "lr": optimizer.param_groups[0]["lr"]}, epoch)
            logger.add_scalars("val", {"loss": avg_val, "perplexity": ppl}, epoch)

        if save and avg_val < best_val:
            best_val = avg_val
            torch.save({"model": model.state_dict(), "tokenizer": tokenizer,
                        "val_loss": best_val, "num_kv_heads": model.num_kv_heads}, save_path)
            print(f"  saved checkpoint ({save_path}, val {avg_val:.4f})")

    if logger is not None:
        logger.close()
    elapsed = max(time.time() - t_start, 1e-9)
    print(f"Training done. Best val loss: {best_val:.4f} "
          f"({len(train_loader) * epochs * toks_per_step / elapsed:,.0f} tok/s)")
    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample(model: GPT, tokenizer: CharTokenizer, prompt: str, n_chars: int, temperature: float,
           top_p: float = 1.0):
    device = next(model.parameters()).device
    model.eval()
    start = tokenizer.encode(prompt)
    idx = torch.tensor([start], dtype=torch.long, device=device)
    out = model.generate(idx, n_chars, temperature=temperature, top_p=top_p)
    print(tokenizer.decode(out[0].tolist()))
