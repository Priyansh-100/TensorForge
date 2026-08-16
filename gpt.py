"""
Mini-GPT: a decoder-only transformer for character-level language modeling.

The insight: for language modeling you don't need an encoder — you only need
the *masked* self-attention part (each token attends to past tokens only).
That's exactly what the decoder half of a seq2seq transformer does, minus
the cross-attention.

Usage:
  python gpt.py --epochs 30 --save
  python gpt.py --load --sample
"""

import argparse
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer import (
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
        # Random start position so every epoch sees different slices
        idx = torch.randint(0, len(self.data) - self.block_size - 1, ())
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
    ).to(device)
    print(f"Model parameters: {model.count_params():,}")

    train_ds = CharDataset(train_data, block_size, num_pairs=20000)
    val_ds = CharDataset(val_data, block_size, num_pairs=2000)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=1000)

    # No padding anywhere → mask is always the causal one
    mask = create_look_ahead_mask(block_size).to(device)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation (no dropout, no grad)
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
            print(f"Epoch {epoch:3d} | train {avg_train:.4f} | val {avg_val:.4f} | perplexity {ppl:.2f}")

        if save and avg_val < best_val:
            best_val = avg_val
            torch.save({"model": model.state_dict(), "tokenizer": tokenizer,
                        "val_loss": best_val, "num_kv_heads": model.num_kv_heads}, save_path)
            print(f"  saved checkpoint ({save_path}, val {avg_val:.4f})")

    print(f"Training done. Best val loss: {best_val:.4f}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--load", action="store_true", help="load gpt.pt and sample")
    parser.add_argument("--save", action="store_true", help="save best checkpoint")
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n-chars", type=int, default=400)
    parser.add_argument("--rope", action="store_true", help="use rotary positional embeddings")
    parser.add_argument("--load-path", type=str, default="gpt.pt", help="checkpoint to load when --load is set")
    parser.add_argument("--save-path", type=str, default="gpt.pt", help="checkpoint to write when --save is set")
    parser.add_argument("--kv-heads", type=int, default=None,
                        help="K/V heads per layer (GQA; must divide --heads). Default: one per query head")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="nucleus sampling mass (default 1.0 = plain temperature sampling)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed RNGs for reproducible training")
    args = parser.parse_args()

    if args.load:
        ckpt = torch.load(args.load_path, map_location="cpu", weights_only=False)
        tokenizer = ckpt["tokenizer"]
        has_learned_pos = "pos_embedding.weight" in ckpt["model"]
        if args.rope and has_learned_pos:
            raise SystemExit(
                f"{args.load_path} was trained with LEARNED position embeddings — "
                f"drop --rope to load it."
            )
        if not args.rope and not has_learned_pos:
            raise SystemExit(
                f"{args.load_path} was trained with RoPE — pass --rope to load it."
            )
        num_kv_heads = args.kv_heads or ckpt.get("num_kv_heads")
        model = GPT(vocab_size=tokenizer.vocab_size, max_len=args.block_size,
                    rope=args.rope, num_kv_heads=num_kv_heads)
        model.load_state_dict(ckpt["model"])
        sample(model, tokenizer, args.prompt, args.n_chars, args.temperature,
               args.top_p)
    else:
        with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()

        tokenizer = CharTokenizer(text)
        print(f"Vocab size: {tokenizer.vocab_size} chars")

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n_val = int(0.1 * len(data))
        train_data, val_data = data[:-n_val], data[-n_val:]

        t0 = time.time()
        model = train(
            tokenizer, train_data, val_data,
            epochs=args.epochs, block_size=args.block_size,
            d_model=args.d_model, num_heads=args.heads, d_ff=args.d_ff,
            num_layers=args.layers, batch_size=args.batch_size, save=args.save,
            rope=args.rope, save_path=args.save_path, num_kv_heads=args.kv_heads,
            seed=args.seed,
        )
        print(f"Trained in {time.time() - t0:.1f}s\n")
        sample(model, tokenizer, args.prompt, args.n_chars, args.temperature)
