#!/usr/bin/env python3
"""
Curriculum learning and progressive training for mini-GPT.

Implements:
- Sequence length curriculum (progressive resizing)
- Data curriculum (easy to hard)
- Learning rate scheduling with curriculum
- Dynamic batch sizing
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.gpt import GPT, CharTokenizer, NoamSchedule
from transformer.bpe import BPETokenizer


@dataclass
class CurriculumStage:
    """Single curriculum stage."""
    name: str
    block_size: int
    batch_size: int
    epochs: int
    lr_multiplier: float = 1.0
    data_fraction: float = 1.0  # Fraction of data to use
    min_loss: float = None  # Stop early if loss below this


DEFAULT_CURRICULUM = [
    CurriculumStage("warmup", block_size=64, batch_size=64, epochs=2, lr_multiplier=0.5),
    CurriculumStage("short", block_size=128, batch_size=32, epochs=3, lr_multiplier=1.0),
    CurriculumStage("medium", block_size=256, batch_size=16, epochs=3, lr_multiplier=0.8),
    CurriculumStage("long", block_size=512, batch_size=8, epochs=2, lr_multiplier=0.5),
]


class CurriculumTrainer:
    """Trainer with curriculum learning support."""
    
    def __init__(self, model, tokenizer, train_data, val_data, curriculum, device, seed=0):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.curriculum = curriculum
        self.device = device
        self.seed = seed
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.stage_history = []
    
    def create_dataloaders(self, block_size, batch_size, data_fraction=1.0):
        """Create dataloaders for current stage."""
        n_train = int(len(self.train_data) * data_fraction)
        n_val = int(len(self.val_data) * data_fraction)
        
        train_subset = self.train_data[:n_train]
        val_subset = self.val_data[:n_val]
        
        def get_batch(data, bs, blk):
            ix = torch.randint(len(data) - blk, (bs,))
            x = torch.stack([data[i:i+blk] for i in ix])
            y = torch.stack([data[i+1:i+blk+1] for i in ix])
            return x.to(self.device), y.to(self.device)
        
        return train_subset, val_subset, lambda bs, blk: get_batch(train_subset, bs, blk)
    
    def train_stage(self, stage, optimizer, lr_scheduler):
        """Train for one curriculum stage."""
        print(f"\n{'='*60}")
        print(f"Stage: {stage.name} | block_size={stage.block_size} | batch_size={stage.batch_size} | epochs={stage.epochs}")
        print(f"{'='*60}")
        
        train_subset, val_subset, get_batch = self.create_dataloaders(
            stage.block_size, stage.batch_size, stage.data_fraction
        )
        
        # Adjust learning rate
        base_lr = optimizer.param_groups[0]['lr']
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * stage.lr_multiplier
        
        stage_start_step = self.global_step
        stage_best_loss = float('inf')
        
        for epoch in range(stage.epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = max(1, len(train_subset) // (stage.batch_size * stage.block_size))
            
            for batch_idx in range(num_batches):
                x, y = get_batch(stage.batch_size, stage.block_size)
                
                # Forward with AMP if available
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = self.model(x)
                        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                elif self.device.type == 'mps':
                    with torch.amp.autocast('mps'):
                        logits = self.model(x)
                        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits = self.model(x)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                
                epoch_loss += loss.item()
                self.global_step += 1
                
                # Log
                if batch_idx % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1}/{stage.epochs} | Batch {batch_idx}/{num_batches} | "
                          f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Step: {self.global_step}")
            
            avg_train_loss = epoch_loss / num_batches
            
            # Validation
            val_loss = self.validate(stage.block_size, stage.batch_size)
            ppl = torch.exp(torch.tensor(val_loss)).item()
            
            print(f"  Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | PPL: {ppl:.4f}")
            
            if val_loss < stage_best_loss:
                stage_best_loss = val_loss
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  *** New best validation loss: {val_loss:.4f} ***")
            
            # Early stopping for stage
            if stage.min_loss and val_loss < stage.min_loss:
                print(f"  Reached target loss {stage.min_loss}, advancing...")
                break
        
        self.stage_history.append({
            "stage": stage.name,
            "block_size": stage.block_size,
            "batch_size": stage.batch_size,
            "epochs_completed": epoch + 1,
            "best_val_loss": stage_best_loss,
            "steps": self.global_step - stage_start_step
        })
        
        return stage_best_loss
    
    def validate(self, block_size, batch_size):
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        num_val_batches = max(1, len(self.val_data) // (batch_size * block_size))
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                ix = torch.randint(len(self.val_data) - block_size, (batch_size,))
                x = torch.stack([self.val_data[i:i+block_size] for i in ix]).to(self.device)
                y = torch.stack([self.val_data[i+1:i+block_size+1] for i in ix]).to(self.device)
                
                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        
        return val_loss / num_val_batches
    
    def run(self, optimizer, lr_scheduler):
        """Run full curriculum training."""
        print(f"Starting curriculum training with {len(self.curriculum)} stages")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Device: {self.device}")
        
        total_start = time.time()
        
        for stage in self.curriculum:
            self.train_stage(stage, optimizer, lr_scheduler)
        
        total_time = time.time() - total_start
        
        print(f"\n{'='*60}")
        print("CURRICULUM TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Total steps: {self.global_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best perplexity: {torch.exp(torch.tensor(self.best_val_loss)).item():.4f}")
        
        print("\nStage summary:")
        for h in self.stage_history:
            print(f"  {h['stage']:10s} | blk={h['block_size']:4d} | bs={h['batch_size']:3d} | "
                  f"epochs={h['epochs_completed']} | val_loss={h['best_val_loss']:.4f} | steps={h['steps']}")
        
        return self.model


def main():
    parser = argparse.ArgumentParser(description="Curriculum learning for mini-GPT")
    parser.add_argument("--curriculum", type=str, default="default", choices=["default", "fast", "thorough"],
                        help="Curriculum preset")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--d-model", type=int, default=384, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of heads")
    parser.add_argument("--d-ff", type=int, default=1536, help="Feedforward dimension")
    parser.add_argument("--rope", action="store_true", default=True, help="Use RoPE")
    parser.add_argument("--no-rope", action="store_false", dest="rope", help="Disable RoPE")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="GQA: number of K/V heads")
    parser.add_argument("--tie-weights", action="store_true", default=True, help="Tie input/output embeddings")
    parser.add_argument("--bpe", action="store_true", help="Use BPE tokenizer")
    parser.add_argument("--bpe-vocab", type=int, default=512, help="BPE vocab size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--save", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    with open(os.path.join(data_dir, "shakespeare.txt"), "r") as f:
        text = f.read()
    
    # Tokenizer
    if args.bpe:
        tokenizer = BPETokenizer(vocab_size=args.bpe_vocab)
        tokenizer.train(text)
        vocab_size = tokenizer.vocab_size
        print(f"BPE tokenizer: {vocab_size} tokens")
    else:
        tokenizer = CharTokenizer(text)
        vocab_size = tokenizer.vocab_size
        print(f"Char tokenizer: {vocab_size} tokens")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    # Model
    model = GPT(
        vocab_size=vocab_size,
        max_len=512,  # Will be resized by curriculum
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope=args.rope,
        num_kv_heads=args.num_kv_heads,
        tie_weights=args.tie_weights,
    ).to(device)
    
    if args.compile:
        model = torch.compile(model)
        print("Using torch.compile")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    # LR scheduler
    lr_scheduler = NoamSchedule(optimizer, d_model=args.d_model, warmup_steps=args.warmup)
    
    # Curriculum
    if args.curriculum == "default":
        curriculum = DEFAULT_CURRICULUM
    elif args.curriculum == "fast":
        curriculum = [
            CurriculumStage("warmup", 64, 64, 1, 0.5),
            CurriculumStage("short", 128, 32, 2, 1.0),
            CurriculumStage("long", 256, 16, 1, 0.5),
        ]
    elif args.curriculum == "thorough":
        curriculum = [
            CurriculumStage("warmup", 32, 128, 3, 0.3),
            CurriculumStage("short", 64, 64, 4, 0.5),
            CurriculumStage("medium", 128, 32, 5, 1.0),
            CurriculumStage("long", 256, 16, 4, 0.8),
            CurriculumStage("xlong", 512, 8, 3, 0.5),
        ]
    
    # Trainer
    trainer = CurriculumTrainer(model, tokenizer, train_data, val_data, curriculum, device, args.seed)
    
    # Train
    model = trainer.run(optimizer, lr_scheduler)
    
    # Save
    if args.save:
        torch.save({
            "model": model.state_dict(),
            "vocab_size": vocab_size,
            "max_len": 512,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "rope": args.rope,
            "num_kv_heads": args.num_kv_heads,
            "tie_weights": args.tie_weights,
            "curriculum": args.curriculum,
            "best_val_loss": trainer.best_val_loss,
        }, args.save)
        print(f"\nSaved checkpoint to {args.save}")


if __name__ == "__main__":
    main()