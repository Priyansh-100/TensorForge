"""
Multi-GPU training with DistributedDataParallel (DDP).

DDP mirrors the model on each GPU, computes gradients on its own shard of the
batch, then averages gradients across ranks via all-reduce before stepping.
Result: effective batch = batch_size * world_size, and each step trains as if
on a single bigger machine.

Run on a multi-GPU box (or CPU smoke test with gloo):
    torchrun --nproc_per_node=2 scripts/train_dist.py --task reverse --epochs 3

    # single process, no torchrun (falls back to plain training):
    python scripts/train_dist.py --task reverse --epochs 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformer.model import Transformer, create_masks, NoamSchedule  # noqa: E402
from transformer.trainer import DATASETS, PAD_TOKEN  # noqa: E402

NUM_GPUS_SIM = 2  # used only by the raw-init fallback for testing


def setup():
    """Initialize the process group. Returns (rank, world_size) or (0, 1)."""
    if "RANK" in os.environ:  # launched via torchrun
        dist.init_process_group(backend="gloo" if not torch.cuda.is_available() else "nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank) if torch.cuda.is_available() else None
        return rank, world_size
    return 0, 1  # plain single-process run


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(task: str, epochs: int):
    rank, world_size = setup()
    num_workers = world_size

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() and num_workers == 1 else "cpu")

    VOCAB_SIZE, SEQ_LEN, D_MODEL, HEADS, D_FF, LAYERS = 20, 5, 64, 4, 128, 2
    BATCH_SIZE = 32
    data_n = 2000

    # Every rank gets the SAME model init (same seed → same weights).
    torch.manual_seed(42)
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL, num_heads=HEADS, d_ff=D_FF,
        num_layers=LAYERS, max_len=SEQ_LEN + 1,
    ).to(device)

    if num_workers > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=None if device.type != "cuda" else [rank])

    # Each epoch, DistributedSampler splits the dataset into world_size shards.
    ds = DATASETS[task](num_samples=data_n, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    sampler = DistributedSampler(ds, num_replicas=num_workers, rank=rank, shuffle=True) if num_workers > 1 else None
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=sampler is None, sampler=sampler)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamSchedule(optimizer, d_model=D_MODEL, warmup_steps=400)

    # Warmup: wait for all ranks to be ready before starting (avoids deadlock).
    if num_workers > 1:
        dist.barrier()

    model.train()
    for epoch in range(1, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)  # reshuffle differently on each rank
        total_loss = 0.0
        n_batches = 0
        for src, tgt_input, tgt_output in loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
            src_mask, tgt_mask = create_masks(src, tgt_input, PAD_TOKEN)

            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        # Aggregate loss across ranks for the log
        if num_workers > 1:
            loss_t = torch.tensor([total_loss / n_batches], device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            loss_t /= num_workers
            avg_loss = loss_t.item()
        else:
            avg_loss = total_loss / n_batches

        if rank == 0 and (epoch % 1 == 0):
            print(f"rank{rank} | epoch {epoch:2d} | world_size {num_workers} | loss {avg_loss:.4f}")

    if rank == 0:
        print(f"Training done on {num_workers} worker(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(DATASETS), default="reverse")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train(args.task, args.epochs)
    cleanup()