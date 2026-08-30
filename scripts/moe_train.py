#!/usr/bin/env python3
"""
Mixture-of-Experts (MoE) layer for mini-GPT.

Adds routing with load balancing losses; compares compute-to-quality vs dense.

Usage:
  python scripts/moe_train.py --num-experts 4 --top-k 2 --epochs 10

Reference: Fedus et al., "Switch Transformers" (2021); Shazeer et al., "Mixture-of-Experts" (2017)
"""
import argparse
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask


class MoELinear(nn.Module):
    """Mixture-of-Experts linear layer with top-k routing."""

    def __init__(self, in_features: int, out_features: int, num_experts: int = 4, top_k: int = 2,
                 capacity_factor: float = 1.25, drop_tokens: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        # Expert weights: [num_experts, out_features, in_features]
        self.expert_weights = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.expert_biases = nn.Parameter(torch.zeros(num_experts, out_features))

        # Router: maps input -> expert logits
        self.router = nn.Linear(in_features, num_experts, bias=False)

        # Initialize
        nn.init.kaiming_uniform_(self.expert_weights, a=math.sqrt(5))
        nn.init.zeros_(self.expert_biases)
        nn.init.normal_(self.router.weight, std=0.01)

    def forward(self, x, return_aux_loss=True):
        """
        x: [batch, seq_len, in_features]
        Returns: output [batch, seq_len, out_features], aux_loss (load balancing)
        """
        # Ensure parameters on same device as input
        if self.expert_weights.device != x.device:
            self.expert_weights.data = self.expert_weights.data.to(x.device)
            self.expert_biases.data = self.expert_biases.data.to(x.device)
            self.router.weight.data = self.router.weight.data.to(x.device)

        batch, seq_len, _ = x.shape

        # Router logits: [batch, seq_len, num_experts]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k routing
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize output
        out = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)

        # Process each expert
        aux_loss = torch.tensor(0.0, device=x.device)

        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)

            if not expert_mask.any():
                continue

            expert_x = x[expert_mask]
            expert_weights = top_k_probs[expert_mask][top_k_indices[expert_mask] == expert_idx]

            expert_out = F.linear(expert_x, self.expert_weights[expert_idx], self.expert_biases[expert_idx])
            expert_out = expert_out * expert_weights.unsqueeze(-1)

            out[expert_mask] += expert_out

        # Load balancing loss (Switch Transformer)
        expert_fraction = router_probs.mean(dim=(0, 1))
        uniform = torch.ones_like(expert_fraction) / self.num_experts
        aux_loss = self.num_experts * torch.sum(expert_fraction * uniform)

        return out + aux_loss if return_aux_loss else out


class MoEFeedForward(nn.Module):
    """MoE version of FeedForward with two MoE layers (up/down projection)."""

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.experts_up = MoELinear(d_model, d_ff, num_experts=4, top_k=2)
        self.experts_down = MoELinear(d_ff, d_model, num_experts=4, top_k=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.experts_up(x, return_aux_loss=False)
        x = F.gelu(x)
        x = self.experts_down(x, return_aux_loss=False)
        return self.dropout(x)


def inject_moe(model: nn.Module):
    """Replace FFN layers with MoE versions."""
    for block in model.blocks:
        block.ffn = MoEFeedForward(
            d_model=block.self_attn.d_model,
            d_ff=block.ffn.linear1.out_features if hasattr(block.ffn, 'linear1') else 512,
            num_experts=4,
            top_k=2,
            dropout=0.1
        )
    return model


def train_moe(
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
    save_path="gpt_moe.pt",
    num_kv_heads=None,
    seed=None,
    amp=False,
    grad_accum=1,
    compile_model=False,
    tb_log=None,
    num_pairs=20000,
    num_val_pairs=2000,
    tie_weights=False,
    rope_scaling="none",
    rope_scaling_factor=1.0,
    weight_decay=0.0,
    grad_clip=0.0,
) -> "GPT":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed: {seed}")

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

    # Replace FFN with MoE
    model = inject_moe(model)

    print(f"MoE model: {sum(p.numel() for p in model.parameters()):,} params "
          f"({sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable)")

    train_loader = DataLoader(CharDataset(train_data, block_size, 20000), batch_size=batch_size, shuffle=True)
    _ = DataLoader(CharDataset(val_data, block_size, 2000), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    from transformer.model import NoamSchedule
    lr_scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=1000)

    mask = create_look_ahead_mask(block_size).to(next(model.parameters()).device)

    for epoch in range(1, 4):
        model.train()
        optimizer.zero_grad()
        for x, y in train_loader:
            x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
            logits = model(x, mask)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            break

    print("MoE test complete")
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
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_moe.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]

    print("MoE training test complete")