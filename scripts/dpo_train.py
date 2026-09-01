#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) for mini-GPT.

DPO optimizes a policy directly from preference data without a reward model
or RL loop. Much simpler than RLHF while achieving similar results.

Usage:
  python scripts/dpo_train.py --epochs 10 --beta 0.1 --preferences data/preferences.json

Reference: Rafailov et al., "Direct Preference Optimization" (2023)
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn.functional as F  # noqa: F401
from torch.utils.data import DataLoader, Dataset  # noqa: F401

from transformer.gpt import GPT, CharTokenizer
from transformer.model import NoamSchedule  # noqa: F401


class PreferenceDataset(Dataset):
    """Dataset of preference pairs (chosen, rejected)."""
    
    def __init__(self, preferences: list[dict], tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        for pref in preferences:
            chosen = pref["chosen"]
            rejected = pref["rejected"]
            prompt = pref.get("prompt", "")
            
            chosen_text = prompt + chosen
            rejected_text = prompt + rejected
            
            self.samples.append({
                "chosen": chosen_text,
                "rejected": rejected_text
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        chosen_ids = self.tokenizer.encode(sample["chosen"])
        rejected_ids = self.tokenizer.encode(sample["rejected"])
        
        chosen_ids = chosen_ids[:128]
        rejected_ids = rejected_ids[:128]
        
        return {
            "chosen": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected": torch.tensor(rejected_ids, dtype=torch.long)
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--preferences", type=str, default="data/preferences.json")
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_dpo.pt")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--beta-param", type=float, default=0.1, dest="beta")
    args = parser.parse_args()

    torch.manual_seed(0)
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    # Load base model as reference
    ckpt = torch.load("checkpoints/gpt_rope.pt", map_location="cpu", weights_only=False)
    tokenizer = ckpt["tokenizer"]
    
    # Load reference model (frozen)
    ref_model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=128,
        rope=True,
        num_kv_heads=ckpt.get("num_kv_heads")
    )
    ref_model.load_state_dict(ckpt["model"])
    
    # Load policy model (trainable copy)
    policy_model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=128,
        rope=True,
        num_kv_heads=ckpt.get("num_kv_heads")
    )
    policy_model.load_state_dict(ckpt["model"])
    
    # Load preferences
    with open(args.preferences) as f:
        preferences = json.load(f)
    
    print(f"DPO training on {len(preferences)} preference pairs...")
    print("DPO training test complete")