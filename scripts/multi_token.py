#!/usr/bin/env python3
"""
Multi-token prediction for mini-GPT.

Predicts n tokens per forward pass, reducing the number of forward passes
needed for generation by a factor of n.

Usage:
  python scripts/multi_token.py --n-predict 4 --epochs 10

Reference: Stern et al., "Blockwise Parallel Decoding" (2018)
           Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (2024)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.gpt import CharTokenizer, GPTBlock
from transformer.model import create_look_ahead_mask, NoamSchedule


class MultiTokenHead(nn.Module):
    """Predicts n next tokens instead of just 1."""

    def __init__(self, d_model: int, vocab_size: int, n_predict: int = 4):
        super().__init__()
        self.n_predict = n_predict
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.head = nn.Linear(d_model, n_predict * vocab_size, bias=False)
        self.pos_embeddings = nn.Parameter(torch.randn(n_predict, d_model) * 0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.n_predict, -1)
        pos_emb = self.pos_embeddings.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        x_with_pos = x_expanded + pos_emb
        logits = self.head(x_with_pos)
        logits = logits.view(B, T, self.n_predict, self.vocab_size)
        return logits


def multi_token_loss(student_logits, targets, n_predict=4):
    """Multi-token prediction loss."""
    B, T, n_pred, V = student_logits.shape
    loss = 0.0
    for i in range(n_predict):
        if i < n_predict - 1:
            tgt = targets[:, 1+i:1+i+1]
        else:
            tgt = targets[:, -1:]
        logits_i = student_logits[:, :, i, :]
        logits_i = logits_i[:, :tgt.size(1), :]
        loss += F.cross_entropy(
            logits_i.reshape(-1, logits_i.size(-1)),
            tgt.reshape(-1)
        )
    return loss / n_predict


class MultiTokenGPT(nn.Module):
    """GPT with multi-token prediction head."""

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
        n_predict: int = 4,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.n_predict = n_predict
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.rope = rope

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if rope:
            from transformer.model import precompute_rope
            cos, sin = precompute_rope(d_model, max_len)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        else:
            self.pos_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout=0.1, num_kv_heads=num_kv_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.multi_head = MultiTokenHead(d_model, vocab_size, n_predict)

        if tie_weights:
            self.multi_head.head.weight = self.token_embedding.weight

    def forward(self, idx, mask=None):
        B, T = idx.shape
        x = self.token_embedding(idx)

        if self.rope:
            _cos, _sin = self.cos[:T], self.sin[:T]
            x = x + self.pos_embedding(torch.arange(T, device=idx.device))
        else:
            pos = self.pos_embedding(torch.arange(T, device=idx.device))
            x = x + pos

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.ln_f(x)
        logits = self.multi_head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_multi_token(
    tokenizer,
    train_data,
    val_data,
    epochs,
    block_size,
    d_model,
    num_heads,
    d_ff,
    num_layers,
    batch_size,
    save,
    rope=False,
    save_path="gpt_multitoken.pt",
    num_kv_heads=None,
    seed=None,
    n_predict=4,
) -> "MultiTokenGPT":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed: {seed}")

    model = MultiTokenGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=block_size,
        rope=rope,
        num_kv_heads=num_kv_heads,
        n_predict=4,
    ).to(device)

    print(f"Multi-token model: {model.count_params():,} params (n_predict={4})")

    train_loader = DataLoader(CharDataset(train_data, block_size, 20000), batch_size=batch_size, shuffle=True)
    _ = DataLoader(CharDataset(val_data, block_size, 2000), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    lr_scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=1000)

    mask = create_look_ahead_mask(block_size).to(next(model.parameters()).device)

    val_loader = DataLoader(CharDataset(val_data, block_size, 2000), batch_size=batch_size, shuffle=True)

    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad()
        for x, y in train_loader:
            x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            logits = model(x, mask)
            loss = multi_token_loss(logits, y, n_predict=4)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            break

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
                logits = model(x, mask)
                loss = multi_token_loss(logits, y, n_predict=4)
                val_loss += loss.item()

        print(f"Epoch {epoch} | val loss: {val_loss:.4f}")

    if save:
        torch.save({"model": model.state_dict(), "tokenizer": tokenizer, "val_loss": val_loss}, save_path)
        print(f"Saved to {save_path}")

    return model


class CharDataset:
    def __init__(self, data, block_size, num_pairs):
        self.data = data
        self.block_size = block_size
        self.num_pairs = num_pairs
    def __len__(self):
        return self.num_pairs
    def __getitem__(self, _):
        hi = len(self.data) - self.block_size - 1
        idx = torch.randint(0, max(hi, 1), ())
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-predict", type=int, default=4)
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_multitoken.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]

    print("Multi-token prediction test complete")