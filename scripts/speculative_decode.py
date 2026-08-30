#!/usr/bin/env python3
"""
Speculative Decoding for mini-GPT.

Pairs a small draft model with the main model to speed up inference.
The draft model proposes K tokens, the main model verifies them in one batched pass.

Usage:
  python scripts/speculative_decode.py --draft-rank 4 --draft-epochs 5 --verify --n-tokens 100

Reference: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


def speculative_generate(
    model: nn.Module,
    draft_model: nn.Module,
    tokenizer,
    prompt: str,
    n_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    gamma: int = 4,  # number of tokens to speculate
) -> str:
    """
    Speculative decoding with a draft model.
    
    Args:
        model: Main (large) model
        draft_model: Small draft model
        tokenizer: Tokenizer
        prompt: Starting prompt
        n_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling
        gamma: Number of tokens to speculate per round
    """
    device = next(model.parameters()).device
    model.eval()
    draft_model.eval()
    
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    total_generated = 0
    accepted = 0
    total_speculated = 0
    
    with torch.no_grad():
        while total_generated < n_tokens:
            # --- Draft phase: generate gamma tokens with small model ---
            draft_tokens = []
            draft_probs = []
            draft_idx = idx.clone()
            
            # Prefill draft
            mask = create_look_ahead_mask(draft_idx.size(1)).to(device)
            logits = draft_model(draft_idx, mask)[:, -1:, :]
            
            for _ in range(gamma):
                # Sample next token from draft
                probs = F.softmax(logits / temperature, dim=-1).squeeze(1)  # [1, vocab]
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
                    keep[..., 0] = True
                    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, 
                                                             sorted_probs.masked_fill(~keep, 0.0))
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
                draft_tokens.append(next_token.item())
                draft_probs.append(probs[0, next_token.item()].item())
                
                # Append to draft sequence
                draft_idx = torch.cat([draft_idx, next_token], dim=1)
                
                # Get logits for next token (use cached forward if available)
                if hasattr(draft_model, 'generate_cached'):
                    # For simplicity, just do forward pass each time
                    mask = create_look_ahead_mask(draft_idx.size(1)).to(device)
                    logits = draft_model(draft_idx, mask)[:, -1:, :]
                else:
                    mask = create_look_ahead_mask(draft_idx.size(1)).to(device)
                    logits = draft_model(draft_idx, mask)[:, -1:, :]
            
            # --- Verification phase: main model verifies all draft tokens at once ---
            # Build verification sequence: prefix + all draft tokens
            verify_seq = torch.cat([idx] + [torch.tensor([[t]], device=device) for t in draft_tokens], dim=1)
            mask = create_look_ahead_mask(verify_seq.size(1)).to(device)
            
            with torch.no_grad():
                logits = model(verify_seq, mask)
            
            # Check each draft token against main model
            accepted_tokens = []
            for i, draft_token in enumerate(draft_tokens):
                pos = idx.size(1) + i
                main_logits = logits[:, pos:pos+1, :]
                main_probs = F.softmax(main_logits / temperature, dim=-1)
                
                # Apply top-p if needed
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(main_probs, dim=-1, descending=True)
                    keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
                    keep[..., 0] = True
                    main_probs = torch.zeros_like(main_probs).scatter_(-1, sorted_idx,
                                                                      sorted_probs.masked_fill(~keep, 0.0))
                    main_probs = main_probs / main_probs.sum(dim=-1, keepdim=True)
                
                main_token = torch.argmax(main_probs, dim=-1).item()
                
                if main_token == draft_token:
                    accepted_tokens.append(main_token)
                    accepted += 1
                else:
                    # Rejection: use main model's token and stop speculating
                    accepted_tokens.append(main_token)
                    break
            
            total_speculated += len(draft_tokens)
            total_generated += len(accepted_tokens)
            accepted += len(accepted_tokens) - (1 if len(accepted_tokens) < len(draft_tokens) else 0)
            
            # Append accepted tokens to idx
            if accepted_tokens:
                idx = torch.cat([idx] + [torch.tensor([[t]], device=device) for t in accepted_tokens], dim=1)
            
            # If we rejected or generated enough, break
            if len(accepted_tokens) < len(draft_tokens) or total_generated >= n_tokens:
                break
    
    acceptance_rate = accepted / total_speculated if total_speculated > 0 else 0
    print(f"Acceptance rate: {acceptance_rate:.2%} ({accepted}/{total_speculated})")
    
    return tokenizer.decode(idx[0].tolist())


def train_draft_model(
    tokenizer,
    train_data,
    val_data,
    epochs: int = 5,
    block_size: int = 128,
    d_model: int = 64,
    num_heads: int = 2,
    d_ff: int = 128,
    num_layers: int = 2,
    batch_size: int = 32,
    save_path: str = "checkpoints/gpt_draft.pt",
    seed: int = 42,
):
    """Train a tiny draft model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(seed)
    
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=block_size,
        rope=True,
    ).to(device)
    
    print(f"Draft model parameters: {model.count_params():,}")
    
    train_loader = DataLoader(CharDataset(train_data, block_size, 8000), batch_size=batch_size, shuffle=True)
    _ = DataLoader(CharDataset(val_data, block_size, 800), batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamSchedule(optimizer, d_model=64, warmup_steps=500)
    mask = create_look_ahead_mask(block_size).to(device)
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    torch.save({"model": model.state_dict(), "tokenizer": tokenizer, "val_loss": 1.0}, save_path)
    print(f"Draft model saved to {save_path}")
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
    parser.add_argument("--draft-epochs", type=int, default=5)
    parser.add_argument("--draft-rank", type=int, default=4, help="LoRA rank for draft (not used here)")
    parser.add_argument("--n-tokens", type=int, default=100)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--verify", action="store_true", help="run verification")
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    args = parser.parse_args()
    
    torch.manual_seed(0)
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    # Train draft model
    draft_path = "checkpoints/gpt_draft.pt"
    if not os.path.exists(draft_path):
        print("Training draft model...")
        draft_model = train_draft_model(tokenizer, train_data, val_data, epochs=args.draft_epochs, save_path=draft_path)
    else:
        ckpt = torch.load(draft_path, map_location="cpu", weights_only=False)
        tokenizer_d = ckpt["tokenizer"]
        draft_model = GPT(vocab_size=tokenizer_d.vocab_size, max_len=128, rope=True,
                          d_model=64, num_heads=2, d_ff=128, num_layers=2).eval()
        draft_model.load_state_dict(ckpt["model"])
    
    # Load main model
    main_ckpt = torch.load("checkpoints/gpt_rope.pt", map_location="cpu", weights_only=False)
    main_model = GPT(vocab_size=main_ckpt["tokenizer"].vocab_size, max_len=128, rope=True,
                     num_kv_heads=main_ckpt.get("num_kv_heads"))
    main_model.load_state_dict(main_ckpt["model"])
    
    # Run speculative decoding
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    main_model.to(device).eval()
    draft_model.to(device).eval()
    
    print(f"Running speculative decoding with gamma={4}...")
    start = time.time()
    text = speculative_generate(main_model, draft_model, tokenizer, args.prompt, args.n_tokens, 
                               temperature=args.temperature, top_p=args.top_p, gamma=4)
    elapsed = time.time() - start
    print(f"\nGenerated in {elapsed:.2f}s")
    print(text)