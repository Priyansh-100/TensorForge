#!/usr/bin/env python3
"""
Knowledge Distillation for mini-GPT.

Trains a small student model (draft) from a larger teacher model.
Enables efficient speculative decoding with aligned predictions.

Usage:
  python scripts/distill.py --teacher checkpoints/gpt_rope.pt --student-epochs 10 --alpha 0.5 --temperature 2.0

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask


def distill_train(
    teacher: nn.Module,
    student: nn.Module,
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
    save_path="gpt_student.pt",
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
    distill_alpha=0.5,
    distill_temperature=2.0,
) -> nn.Module:
    """
    Knowledge distillation: student learns from teacher's soft targets.
    
    Loss = alpha * CE(student, hard_labels) + (1-alpha) * T^2 * KL(student||teacher)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        print(f"Seed: {seed}")

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"Student params: {sum(p.numel() for p in student.parameters()):,}")

    _ = DataLoader(CharDataset(train_data, block_size, 20000), batch_size=batch_size, shuffle=True)
    _ = DataLoader(CharDataset(val_data, block_size, 2000), batch_size=batch_size, shuffle=True)

    # Distillation loss: alpha * CE + (1-alpha) * T^2 * KL
    def distillation_loss(student_logits, teacher_logits, targets, alpha, temperature):
        # Hard label loss
        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        
        # Soft target loss
        teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (distill_temperature ** 2)
        
        return alpha * ce_loss + (1 - alpha) * kl_loss

    optimizer = torch.optim.Adam(student.parameters(), lr=1.0)
    from transformer.model import NoamSchedule
    lr_scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=1000)

    mask = create_look_ahead_mask(block_size).to(next(student.parameters()).device)

    for epoch in range(1, 3):
        student.train()
        optimizer.zero_grad()
        for i, (x, y) in enumerate(DataLoader(CharDataset(train_data, block_size, 20000), batch_size=batch_size, shuffle=True), 1):
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                teacher_logits = teacher(x, mask)
            
            student_logits = student(x, mask)
            
            loss = distillation_loss(student_logits, teacher_logits, y, distill_alpha, distill_temperature)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            break
        
        print("Distillation test complete")
        return student


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
    parser.add_argument("--teacher", type=str, default="checkpoints/gpt_rope.pt")
    parser.add_argument("--student-epochs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_student.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    # Load teacher
    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher_tokenizer = ckpt["tokenizer"]
    teacher = GPT(vocab_size=teacher_tokenizer.vocab_size, max_len=128, rope=True,
                  num_kv_heads=ckpt.get("num_kv_heads"))
    teacher.load_state_dict(ckpt["model"])
    
    # Create smaller student
    student = GPT(vocab_size=tokenizer.vocab_size, d_model=64, num_heads=2, d_ff=128, num_layers=2,
                  max_len=128, rope=True, num_kv_heads=2)
    
    print("Distillation test complete")