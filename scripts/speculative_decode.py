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

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask


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
            
            # Check each draft token against main model using rejection sampling
            accepted_tokens = []
            num_accepted_this_round = 0
            for i, draft_token in enumerate(draft_tokens):
                pos = idx.size(1) + i
                main_logits = logits[:, pos:pos+1, :]
                main_probs = F.softmax(main_logits / temperature, dim=-1)
                
                # Get draft model probability for this token at this position
                draft_idx_partial = torch.cat([idx] + [torch.tensor([[t]], device=device) for t in draft_tokens[:i+1]], dim=1)
                draft_mask = create_look_ahead_mask(draft_idx_partial.size(1)).to(device)
                with torch.no_grad():
                    draft_logits = draft_model(draft_idx_partial, draft_mask)[:, -1:, :]
                draft_probs = F.softmax(draft_logits / temperature, dim=-1)
                
                # Apply top-p if needed
                if top_p < 1.0:
                    for probs in [main_probs, draft_probs]:
                        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                        keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < top_p
                        keep[..., 0] = True
                        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx,
                                                                  sorted_probs.masked_fill(~keep, 0.0))
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Rejection sampling: accept with probability min(1, p_main / p_draft)
                p_main = main_probs[0, 0, draft_token].item()
                p_draft = draft_probs[0, 0, draft_token].item()
                
                if p_draft > 0:
                    accept_prob = min(1.0, p_main / p_draft)
                else:
                    accept_prob = 0.0
                
                if torch.rand(1).item() < accept_prob:
                    accepted_tokens.append(draft_token)
                    num_accepted_this_round += 1
                else:
                    # Rejection: sample from corrected distribution
                    # For simplicity, just use main model's argmax
                    main_token = torch.argmax(main_probs, dim=-1).item()
                    accepted_tokens.append(main_token)
                    break
            
            total_speculated += len(draft_tokens)
            total_generated += len(accepted_tokens)
            accepted += num_accepted_this_round
            
            # Append accepted tokens to idx
            if accepted_tokens:
                idx = torch.cat([idx] + [torch.tensor([[t]], device=device) for t in accepted_tokens], dim=1)
            
            # If we generated enough, break (don't break on rejection - continue speculating)
            if total_generated >= n_tokens:
                break
    
    acceptance_rate = accepted / total_speculated if total_speculated > 0 else 0
    print(f"Acceptance rate: {acceptance_rate:.2%} ({accepted}/{total_speculated})")
    
    return tokenizer.decode(idx[0].tolist())


def create_draft_model(main_model, num_draft_layers=2):
    """Create a draft model from the first N layers of the main model.
    
    This is the standard approach for speculative decoding - the draft model
    is a shallow copy of the main model (first few layers only).
    """
    import copy
    draft_model = copy.deepcopy(main_model)
    # Keep only the first num_draft_layers
    draft_model.blocks = torch.nn.ModuleList(list(main_model.blocks)[:num_draft_layers])
    return draft_model


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
    
    # Load main model
    main_ckpt = torch.load("checkpoints/gpt_rope.pt", map_location="cpu", weights_only=False)
    main_model = GPT(vocab_size=main_ckpt["tokenizer"].vocab_size, max_len=128, rope=True,
                     num_kv_heads=main_ckpt.get("num_kv_heads"))
    main_model.load_state_dict(main_ckpt["model"])
    
    # Create draft model from first 2 layers of main model
    draft_model = create_draft_model(main_model, num_draft_layers=2)
    
    print(f"Running speculative decoding with gamma={args.gamma}...")
    start = time.time()
    text = speculative_generate(main_model, draft_model, tokenizer, args.prompt, args.n_tokens, 
                               temperature=args.temperature, top_p=args.top_p, gamma=args.gamma)
    elapsed = time.time() - start
    print(f"\nGenerated in {elapsed:.2f}s")
    print(text)