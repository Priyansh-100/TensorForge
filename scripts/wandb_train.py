#!/usr/bin/env python3
"""
Weights & Biases integration for mini-GPT experiment tracking.

Usage:
  python scripts/wandb_train.py --project minigpt --epochs 30 --wandb
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


def train_with_wandb(
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
    save_path="gpt_wandb.pt",
    num_kv_heads=None,
    seed=None,
    amp=False,
    grad_accum=1,
    compile_model=False,
    tie_weights=False,
    project="minigpt",
    run_name=None,
    tags=None,
    config=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

    # Initialize W&B
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: pip install wandb")
        return None

    # Default config
    default_config = {
        "architecture": "GPT",
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "rope": rope,
        "num_kv_heads": num_kv_heads,
        "tie_weights": tie_weights,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "epochs": epochs,
        "amp": amp,
        "compile": compile_model,
        "seed": seed,
        "device": str(device),
    }
    if config:
        default_config.update(config)

    wandb.init(
        project=project,
        name=run_name,
        config=default_config,
        tags=tags or ["mini-gpt", "shakespeare"],
    )

    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=block_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        rope=rope,
        num_kv_heads=num_kv_heads,
        tie_weights=tie_weights,
    ).to(device)

    if compile_model:
        model = torch.compile(model)
        print("Using torch.compile")

    print(f"Model parameters: {model.count_params():,}")

    # Watch model gradients
    wandb.watch(model, log="all", log_freq=100)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=500)
    scaler = torch.amp.GradScaler('cuda') if amp and device.type == 'cuda' else None
    mps_scaler = torch.amp.GradScaler('mps') if amp and device.type == 'mps' else None

    mask = create_look_ahead_mask(block_size).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            if amp and (scaler or mps_scaler):
                ctx = torch.amp.autocast(device.type)
                with ctx:
                    logits = model(x, mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum

                current_scaler = scaler if device.type == 'cuda' else mps_scaler
                current_scaler.scale(loss).backward()

                if (batch_idx + 1) % grad_accum == 0:
                    current_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    current_scaler.step(optimizer)
                    current_scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log to W&B
                    wandb.log({
                        "train/loss": loss.item() * grad_accum,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/step": global_step,
                        "train/epoch": epoch,
                    })
            else:
                logits = model(x, mask)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum
                loss.backward()

                if (batch_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    wandb.log({
                        "train/loss": loss.item() * grad_accum,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/step": global_step,
                        "train/epoch": epoch,
                    })

            epoch_loss += loss.item() * grad_accum * y.numel()
            epoch_tokens += y.numel()

        avg_train_loss = epoch_loss / epoch_tokens
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if amp and (scaler or mps_scaler):
                    ctx = torch.amp.autocast(device.type)
                    with ctx:
                        logits = model(x, mask)
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits = model(x, mask)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                val_loss += loss.item() * y.numel()
                val_tokens += y.numel()

        avg_val_loss = val_loss / val_tokens
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "train/epoch_ppl": train_ppl,
            "val/loss": avg_val_loss,
            "val/ppl": val_ppl,
            "val/best_loss": min(best_val_loss, avg_val_loss),
        })

        print(f"Epoch {epoch:3d} | train loss: {avg_train_loss:.4f} | train ppl: {train_ppl:.2f} | "
              f"val loss: {avg_val_loss:.4f} | val ppl: {val_ppl:.2f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save:
                torch.save({
                    "model": model.state_dict(),
                    "tokenizer": tokenizer,
                    "val_loss": best_val_loss,
                    "config": default_config,
                }, save_path)
                print(f"  saved checkpoint (val {best_val_loss:.4f})")
                # Log artifact
                artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

    wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--no-rope", action="store_false", dest="rope")
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_wandb.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--project", type=str, default="minigpt")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tags", type=str, nargs="+", default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()

    if not args.wandb:
        print("W&B logging not enabled. Use --wandb flag.")
        sys.exit(0)

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]

    train_with_wandb(
        tokenizer, train_data, val_data,
        epochs=args.epochs,
        block_size=args.block_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        save=args.save,
        rope=args.rope,
        save_path=args.save_path,
        num_kv_heads=args.kv_heads,
        seed=args.seed,
        amp=args.amp,
        grad_accum=args.grad_accum,
        compile_model=args.compile,
        tie_weights=args.tie_weights,
        project=args.project,
        run_name=args.run_name,
        tags=args.tags,
    )