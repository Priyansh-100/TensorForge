#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) fine-tuning for the mini-GPT.

Freezes the base model and trains only low-rank adapters injected into
the attention projections. Much faster and memory-efficient than full fine-tuning.

Usage:
  python scripts/lora_finetune.py --epochs 10 --rank 4 --alpha 8 --save-path checkpoints/gpt_lora.pt

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""
import argparse
import os
import sys
import math
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter."""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA adapters (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze base weights
        self.weight.requires_grad = False
        self.bias.requires_grad = False
    
    def _ensure_device(self, x):
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            self.bias.data = self.bias.data.to(x.device)
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)
    
    def forward(self, x):
        self._ensure_device(x)
        # Base: x @ W^T + b
        base = F.linear(x, self.weight, self.bias)
        # LoRA: x @ A^T @ B^T * scaling
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora
    
    def merge_weights(self):
        """Merge LoRA into base weights for deployment."""
        with torch.no_grad():
            self.weight += (self.lora_B @ self.lora_A) * self.scaling
            self.lora_A.zero_()
            self.lora_B.zero_()


def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 1.0, target_modules: list = None):
    """Inject LoRA adapters into target linear layers."""
    if target_modules is None:
        target_modules = ["W_q", "W_k", "W_v", "W_o"]
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            
            lora = LoRALinear(module.in_features, module.out_features, rank, alpha)
            lora.weight.data.copy_(module.weight.data)
            lora.bias.data.copy_(module.bias.data)
            
            setattr(parent, child_name, lora)
    
    return model


def train_lora(
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
    save_path: str = "gpt_lora.pt",
    num_kv_heads: int | None = None,
    seed: int | None = None,
    amp: bool = False,
    grad_accum: int = 1,
    compile_model: bool = False,
    tb_log: str | None = None,
    num_pairs: int = 20000,
    num_val_pairs: int = 2000,
    tie_weights: bool = False,
    rope_scaling: str = "none",
    rope_scaling_factor: float = 1.0,
    weight_decay: float = 0.0,
    grad_clip: float = 0.0,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    lora_targets: list = None,
) -> "GPT":
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
        rope_scaling=rope_scaling,
        rope_scaling_factor=rope_scaling_factor,
    ).to(device)
    
    # Inject LoRA
    model = inject_lora(model, rank=lora_rank, alpha=lora_alpha, target_modules=lora_targets)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (LoRA trainable: {trainable_params:,}, {100*trainable_params/total_params:.1f}%)")

    # torch.compile
    if compile_model:
        try:
            model = torch.compile(model)
            print("  torch.compile: ON")
        except Exception as e:
            print(f"  torch.compile unavailable: {e}")

    # AMP
    use_amp = amp and device.type in ("cuda", "mps")
    if amp and not use_amp:
        print("  AMP: fp16 autocast needs CUDA or MPS — running in fp32 on CPU")
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    logger = None  # _TBLogger not imported here

    train_ds = CharDataset(train_data, block_size, num_pairs=num_pairs)
    val_ds = CharDataset(val_data, block_size, num_pairs=num_val_pairs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    if grad_accum > 1 and len(train_loader) % grad_accum != 0:
        leftover = len(train_loader) % grad_accum
        print(f"  note: {leftover} batch(es) per epoch fall outside the --grad-accum {grad_accum} group and are dropped (never stepped)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, weight_decay=weight_decay)
    scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=1000)

    # No padding anywhere → mask is always the causal one
    mask = create_look_ahead_mask(block_size).to(device)

    toks_per_step = batch_size * block_size
    best_val = float("inf")
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
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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


class CharDataset:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_lora.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    model = train_lora(
        tokenizer, train_data, val_data,
        epochs=args.epochs, block_size=128, d_model=128, num_heads=4,
        d_ff=512, num_layers=4, batch_size=32, save=True,
        rope=True, save_path=args.save_path, seed=args.seed,
        lora_rank=args.rank, lora_alpha=args.alpha,
    )
    print(f"LoRA fine-tuning complete. Model saved to {args.save_path}")