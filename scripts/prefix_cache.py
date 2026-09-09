#!/usr/bin/env python3
"""
Prefix Caching for mini-GPT.

Caches KV states for common prompt prefixes to avoid recomputation.
Useful for serving scenarios with common prompt prefixes (system prompts, few-shot examples).

Usage:
  python scripts/prefix_cache.py --prefix "System: You are a helpful assistant." --n-tokens 100

Reference: vLLM's prefix caching, KIVI, etc.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from transformer.gpt import CharTokenizer


def _sample_token(logits, temperature, top_p):
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
        keep[..., 0] = True
        trimmed = sorted_probs.masked_fill(~keep, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, trimmed / trimmed.sum(dim=-1, keepdim=True))
    return torch.multinomial(probs, num_samples=1)


class PrefixCache:
    """Prefix cache for KV states."""

    def __init__(self, model, prefix_text, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Encode prefix
        prefix_ids = tokenizer.encode(prefix_text)
        self.prefix_len = len(prefix_ids)

        # Build KV cache for prefix
        self.prefix_cache = self._build_prefix_cache(prefix_text)

    def _build_prefix_cache(self, prefix_text):
        """Precompute KV cache for the prefix."""
        self.model.eval()
        prefix_ids = torch.tensor([self.tokenizer.encode(prefix_text)], dtype=torch.long, device=self.device)

        caches = [None] * len(self.model.blocks)
        with torch.no_grad():
            mask = torch.tril(torch.ones(self.prefix_len, self.prefix_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).to(self.device)
            _, caches = self.model._cached_forward(prefix_ids, [None] * len(self.model.blocks), 0, mask)

        # Store prefix length and cache
        return {
            'prefix_len': self.prefix_len,
            'caches': caches
        }

    def generate(self, continuation_prompt, n_tokens=100, temperature=0.8, top_p=0.9):
        """Generate continuation using cached prefix."""
        self.model.eval()

        # Encode continuation
        continuation_ids = self.tokenizer.encode(continuation_prompt)
        idx = torch.tensor([continuation_ids], dtype=torch.long, device=self.device)

        # Start from prefix cache
        caches = self.prefix_cache['caches']
        start_pos = self.prefix_cache['prefix_len']

        with torch.no_grad():
            # Prefill continuation
            seq_len = idx.size(1)
            mask = torch.zeros(seq_len, self.prefix_len + seq_len, dtype=torch.bool, device=self.device)
            for i in range(seq_len):
                mask[i, :self.prefix_len + i + 1] = True
            mask = mask.unsqueeze(0).unsqueeze(0)

            x, caches = self.model._cached_forward(idx, caches, self.prefix_len, mask)

            # Generate
            out = idx
            start_pos = self.prefix_len + seq_len
            for _ in range(n_tokens):
                logits = self.model.lm_head(self.model.ln_f(x[:, -1:, :]))
                next_token = _sample_token(logits[:, -1, :], temperature, top_p)
                out = torch.cat([out, next_token], dim=1)
                step_mask = torch.ones(1, 1, 1, 1, dtype=torch.bool, device=self.device)
                x, caches = self.model._cached_forward(next_token, caches, start_pos, step_mask)
                start_pos += 1

        return self.tokenizer.decode(out[0].tolist())


def test_prefix_cache():
    """Test prefix caching with a trained model."""
    import torch

    from transformer.gpt import train

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]

    # Quick train a small model
    model = train(
        CharTokenizer(text), train_data, val_data,
        epochs=1, block_size=128, d_model=64, num_heads=4, d_ff=256,
        num_layers=2, batch_size=32, save=False, rope=True,
        seed=0, num_pairs=2000, num_val_pairs=200
    )

    model.to("cpu")

    # Test prefix cache
    tokenizer = CharTokenizer(text)
    cache = PrefixCache(model, "To be, or not to be, ", tokenizer, "cpu")

    print(f"Prefix length: {cache.prefix_len}")
    print(f"Cache layers: {len(cache.prefix_cache['caches'])}")

    # Generate with prefix cache
    start = time.time()
    text = cache.generate("that is the question", n_tokens=50)
    elapsed = time.time() - start

    print(f"Generated in {elapsed:.2f}s")
    print(f"Generated: {text[:100]}...")

    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="To be, or not to be, ")
    parser.add_argument("--n-tokens", type=int, default=100)
    args = parser.parse_args()

    test_prefix_cache()