#!/usr/bin/env python3
"""
Hyperparameter optimization for mini-GPT using Optuna.

Usage:
  python scripts/hpo.py --n-trials 20 --epochs 5 --study-name minigpt-hpo
"""

import argparse
import os
import sys
import json

try:
    import optuna
except ImportError:
    optuna = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


def objective(trial, tokenizer, train_data, val_data, epochs, block_size, device, base_config):
    """Optuna objective function."""
    
    # Suggest hyperparameters
    d_model = trial.suggest_categorical("d_model", [128, 256, 384, 512])
    num_heads = trial.suggest_categorical("num_heads", [4, 6, 8])
    num_layers = trial.suggest_categorical("num_layers", [4, 6, 8])
    d_ff = trial.suggest_categorical("d_ff", [512, 1024, 2048, 3072])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2, 4])
    
    # Ensure d_model divisible by num_heads
    if d_model % num_heads != 0:
        raise optuna.TrialPruned()
    
    # Effective batch size constraint
    effective_bs = batch_size * grad_accum
    if effective_bs > 512:
        raise optuna.TrialPruned()
    
    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=block_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        dropout=dropout,
        rope=True,
    ).to(device)
    
    print(f"Trial {trial.number}: d_model={d_model}, heads={num_heads}, layers={num_layers}, "
          f"d_ff={d_ff}, bs={batch_size}, lr={lr:.2e}, wd={weight_decay:.2e}")
    
    # Data
    from transformer.gpt import CharDataset
    train_loader = DataLoader(
        CharDataset(train_data, block_size, 20000), 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        CharDataset(val_data, block_size, 2000), 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=500)
    mask = create_look_ahead_mask(block_size).to(device)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            logits = model(x, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / grad_accum
            loss.backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, mask)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item() * y.numel()
                val_tokens += y.numel()
        
        avg_val_loss = val_loss / val_tokens
        
        # Report intermediate value for pruning
        trial.report(avg_val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"  Epoch {epoch}: val_loss={avg_val_loss:.4f}, best={best_val_loss:.4f}")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--study-name", type=str, default="minigpt-hpo")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///hpo.db)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="hpo_results.json")
    args = parser.parse_args()
    
    try:
        import optuna
    except ImportError:
        print("optuna not installed. Run: pip install optuna")
        return
    
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, tokenizer, train_data, val_data, 
                               args.epochs, args.block_size, device, {}),
        n_trials=args.n_trials,
        timeout=None,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_loss: {study.best_trial.value:.4f}")
    print("Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print top 5 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))
    print("\nTop 5 trials:")
    for i, t in enumerate(sorted_trials[:5]):
        if t.value is not None:
            print(f"  {i+1}. Trial {t.number}: val_loss={t.value:.4f}, params={t.params}")


if __name__ == "__main__":
    main()