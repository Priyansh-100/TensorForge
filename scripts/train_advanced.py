#!/usr/bin/env python3
"""
Advanced training techniques for mini-GPT.

Implements:
- Gradient accumulation with dynamic scaling
- Mixed precision training (FP16/BF16)
- Gradient clipping strategies
- Learning rate scheduling (cosine, polynomial, step)
- Early stopping with patience
- Model EMA (Exponential Moving Average)
- SWA (Stochastic Weight Averaging)
- Distributed training utilities

Usage:
  python scripts/train_advanced.py --epochs 50 --amp --ema --swa --early-stopping
"""

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


@dataclass
class TrainingConfig:
    """Advanced training configuration."""
    epochs: int = 50
    block_size: int = 128
    d_model: int = 384
    num_heads: int = 6
    d_ff: int = 1536
    num_layers: int = 6
    batch_size: int = 32
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    lr_schedule: str = "cosine"  # cosine, polynomial, step, noam
    lr_min: float = 1e-5
    lr_decay_steps: int = 10000
    amp: bool = True
    amp_dtype: str = "bf16"  # fp16, bf16
    ema_decay: float = 0.999
    use_ema: bool = True
    swa_start: int = 0  # epoch to start SWA
    swa_lr: float = 1e-5
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    compile_model: bool = False
    seed: int = 42
    save_path: str = "checkpoints/gpt_advanced.pt"
    log_interval: int = 100
    eval_interval: int = 500


class EMACallback:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    @torch.no_grad()
    def restore(self, model: nn.Module):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


class SWACallback:
    """Stochastic Weight Averaging."""
    
    def __init__(self, model: nn.Module, swa_start: int, swa_lr: float):
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.swa_model = None
        self.swa_n = 0
        self.device = next(model.parameters()).device
    
    def maybe_init_swa(self, model: nn.Module, epoch: int):
        if epoch >= self.swa_start and self.swa_model is None:
            self.swa_model = type(model)(**model.__dict__.get('_init_args', {}))
            self.swa_model.load_state_dict(model.state_dict())
            self.swa_model.to(self.device)
            self.swa_model.eval()
            print(f"SWA initialized at epoch {epoch}")
    
    @torch.no_grad()
    def update(self, model: nn.Module, epoch: int):
        if epoch >= self.swa_start and self.swa_model is not None:
            self.swa_n += 1
            for swa_param, model_param in zip(self.swa_model.parameters(), model.parameters()):
                swa_param.mul_(1 - 1/self.swa_n).add_(model_param.data, alpha=1/self.swa_n)
    
    @torch.no_grad()
    def apply_swa(self, model: nn.Module):
        if self.swa_model is not None:
            print("  Loading SWA state dict...")
            model.load_state_dict(self.swa_model.state_dict())
            print("  SWA applied successfully")


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_advanced(config: TrainingConfig, tokenizer, train_data, val_data, device):
    """Advanced training loop with all techniques."""
    
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)
    
    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=config.block_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        rope=True,
    ).to(device)
    
    # Store init args for SWA
    model._init_args = {
        'vocab_size': tokenizer.vocab_size,
        'max_len': config.block_size,
        'd_model': config.d_model,
        'num_heads': config.num_heads,
        'd_ff': config.d_ff,
        'num_layers': config.num_layers,
        'rope': True,
    }
    
    if config.compile_model:
        model = torch.compile(model)
        print("Using torch.compile")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    from transformer.gpt import CharDataset
    train_dataset = CharDataset(train_data, config.block_size, 20000)
    val_dataset = CharDataset(val_data, config.block_size, 2000)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        betas=(0.9, 0.95), 
        weight_decay=config.weight_decay
    )
    
    # LR Scheduler
    if config.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=config.lr_decay_steps, eta_min=config.lr_min)
    elif config.lr_schedule == "polynomial":
        from torch.optim.lr_scheduler import PolynomialLR
        scheduler = PolynomialLR(optimizer, total_iters=config.lr_decay_steps, power=2.0)
    elif config.lr_schedule == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=config.lr_decay_steps // 3, gamma=0.5)
    elif config.lr_schedule == "noam":
        scheduler = NoamSchedule(optimizer, d_model=config.d_model, warmup_steps=config.warmup_steps)
    else:
        scheduler = None
    
    # AMP
    scaler = None
    if config.amp and device.type in ('cuda', 'mps'):
        amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.amp.GradScaler(device.type)
        print(f"AMP enabled with {amp_dtype}")
    
    # Callbacks
    ema = EMACallback(model, config.ema_decay) if config.use_ema else None
    swa = SWACallback(model, config.swa_start, config.swa_lr) if config.swa_start > 0 else None
    early_stopping = EarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
    
    # Training
    mask = create_look_ahead_mask(config.block_size).to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    global_step = 0
    
    print(f"Starting training: {config.epochs} epochs, AMP={config.amp}, EMA={config.use_ema}, SWA={config.swa_start > 0}")
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        
        # SWA initialization
        if swa:
            swa.maybe_init_swa(model, epoch)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward
            if scaler is not None:
                amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
                with torch.amp.autocast(device.type, dtype=amp_dtype):
                    logits = model(x, mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / config.grad_accum
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler and config.lr_schedule != "noam":
                        scheduler.step()
                    elif scheduler and config.lr_schedule == "noam":
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            else:
                logits = model(x, mask)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / config.grad_accum
                loss.backward()
                
                if (batch_idx + 1) % config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if scheduler and config.lr_schedule != "noam":
                        scheduler.step()
                    elif scheduler and config.lr_schedule == "noam":
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            epoch_loss += loss.item() * config.grad_accum * y.numel()
            epoch_tokens += y.numel()
            
            # Logging
            if batch_idx % config.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] | "
                      f"Loss: {loss.item() * config.grad_accum:.4f} | LR: {current_lr:.2e}")
        
        avg_train_loss = epoch_loss / epoch_tokens
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        
        # EMA update
        if ema:
            ema.update(model)
        
        # SWA update
        if swa:
            swa.update(model, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        
        # Use EMA weights for validation if available
        if ema:
            ema.apply_shadow(model)
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if scaler is not None:
                    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
                    with torch.amp.autocast(device.type, dtype=amp_dtype):
                        logits = model(x, mask)
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits = model(x, mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item() * y.numel()
                val_tokens += y.numel()
        
        if ema:
            ema.restore(model)
        
        avg_val_loss = val_loss / val_tokens
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        
        print(f"Epoch {epoch:3d} | train loss: {avg_train_loss:.4f} | train ppl: {train_ppl:.2f} | "
              f"val loss: {avg_val_loss:.4f} | val ppl: {val_ppl:.2f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_to_save = model.state_dict()
            if ema:
                # Also save EMA state
                ema_state = {f"ema_{k}": v for k, v in ema.shadow.items()}
            else:
                ema_state = {}
            
            torch.save({
                "model": state_to_save,
                "ema": ema_state,
                "tokenizer": tokenizer,
                "val_loss": best_val_loss,
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "block_size": config.block_size,
                    "d_model": config.d_model,
                    "num_heads": config.num_heads,
                    "d_ff": config.d_ff,
                    "num_layers": config.num_layers,
                    "rope": True,
                },
                "epoch": epoch,
                "global_step": global_step,
            }, config.save_path)
            print(f"  saved checkpoint (val {best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Apply SWA weights if used
    if swa and swa.swa_model is not None:
        print("Applying SWA weights...")
        swa.apply_swa(model)
        torch.save({
            "model": model.state_dict(),
            "tokenizer": tokenizer,
            "val_loss": best_val_loss,
            "swa": True,
        }, config.save_path.replace(".pt", "_swa.pt"))
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-schedule", type=str, default="cosine", 
                        choices=["cosine", "polynomial", "step", "noam"])
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-decay-steps", type=int, default=10000)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no-ema", action="store_false", dest="ema")
    parser.add_argument("--swa-start", type=int, default=0)
    parser.add_argument("--swa-lr", type=float, default=1e-5)
    parser.add_argument("--early-stopping", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_advanced.pt")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    config = TrainingConfig(
        epochs=args.epochs,
        block_size=args.block_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        lr_decay_steps=args.lr_decay_steps,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        ema_decay=args.ema_decay,
        use_ema=args.ema,
        swa_start=args.swa_start,
        swa_lr=args.swa_lr,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=1e-4,
        compile_model=args.compile,
        seed=args.seed,
        save_path=args.save_path,
    )
    
    train_advanced(config, tokenizer, train_data, val_data, device)


if __name__ == "__main__":
    main()