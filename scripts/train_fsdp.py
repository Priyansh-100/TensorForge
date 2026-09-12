#!/usr/bin/env python3
"""
FSDP (Fully Sharded Data Parallel) training for mini-GPT.

Enables training models larger than single-GPU memory by sharding
optimizer states, gradients, and parameters across GPUs.

Usage:
  torchrun --nproc_per_node=4 scripts/train_fsdp.py --epochs 10 --sharding-strategy FULL_SHARD
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
import functools

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


def setup_fsdp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None


def cleanup():
    dist.destroy_process_group()


def train_fsdp(
    rank,
    world_size,
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
    rope=False,
    num_kv_heads=None,
    sharding_strategy="FULL_SHARD",
    mixed_precision=True,
    cpu_offload=False,
    save_path="checkpoints/gpt_fsdp.pt",
    seed=42,
):
    """FSDP training function."""
    setup_fsdp(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    if seed is not None:
        torch.manual_seed(seed + rank)
    
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
    ).to(device)
    
    # FSDP wrapping policy
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={type(model.blocks[0])},
    )
    
    # Mixed precision
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            reduce_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            buffer_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    
    # CPU offload
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None
    
    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
    }
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_config,
        sharding_strategy=strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=rank if torch.cuda.is_available() else None,
        sync_module_states=True,
    )
    
    # Data
    from transformer.gpt import CharDataset
    train_dataset = CharDataset(train_data, block_size, 20000)
    val_dataset = CharDataset(val_data, block_size, 2000)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = NoamSchedule(optimizer, d_model=d_model, warmup_steps=500)
    criterion = nn.CrossEntropyLoss()
    mask = create_look_ahead_mask(block_size).to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0.0
        epoch_tokens = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x, mask)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() * y.numel()
            epoch_tokens += y.numel()
        
        avg_train_loss = epoch_loss / epoch_tokens
        
        # Validation (only rank 0 prints)
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
        
        # All-reduce for logging
        train_loss_tensor = torch.tensor(avg_train_loss, device=device)
        val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        
        if rank == 0:
            train_ppl = torch.exp(train_loss_tensor).item()
            val_ppl = torch.exp(val_loss_tensor).item()
            print(f"Epoch {epoch:3d} | train loss: {train_loss_tensor:.4f} | train ppl: {train_ppl:.2f} | "
                  f"val loss: {val_loss_tensor:.4f} | val ppl: {val_ppl:.2f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")
            
            if val_loss_tensor < best_val_loss:
                best_val_loss = val_loss_tensor
                # Save checkpoint (only rank 0 saves)
                if save_path:
                    # Need to gather full state dict
                    state_dict = model.state_dict()
                    # Only save on rank 0
                    torch.save({
                        "model": state_dict,
                        "tokenizer": tokenizer,
                        "val_loss": best_val_loss.item(),
                        "config": {
                            "vocab_size": tokenizer.vocab_size,
                            "block_size": block_size,
                            "d_model": d_model,
                            "num_heads": num_heads,
                            "d_ff": d_ff,
                            "num_layers": num_layers,
                            "rope": rope,
                            "num_kv_heads": num_kv_heads,
                        }
                    }, save_path)
                    print(f"  saved checkpoint (val {best_val_loss:.4f})")
    
    cleanup()
    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument("--sharding-strategy", type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"])
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--save-path", type=str, default="checkpoints/gpt_fsdp.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Check if running under torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("This script must be run with torchrun:")
        print("  torchrun --nproc_per_node=4 scripts/train_fsdp.py --epochs 10")
        return
    
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]
    
    train_fsdp(
        rank, world_size, tokenizer, train_data, val_data,
        epochs=args.epochs,
        block_size=args.block_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        rope=args.rope,
        num_kv_heads=args.kv_heads,
        sharding_strategy=args.sharding_strategy,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        save_path=args.save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()