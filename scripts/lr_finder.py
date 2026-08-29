#!/usr/bin/env python3
"""
Learning rate finder / LR curve visualizer.

Plots the Noam schedule and CosineWarmRestarts schedules to help
pick warmup_steps, T_0, T_mult, eta_min.

Usage:
  python scripts/lr_finder.py --d-model 128 --warmup 1000 --steps 5000
  python scripts/lr_finder.py --scheduler cosine_restarts --t0 1000 --t-mult 2 --eta-min 1e-5 --steps 5000

Output: plots/lr_curve.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn

from transformer.model import NoamSchedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_noam(d_model: int, warmup: int, total_steps: int, out_path: str):
    optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = NoamSchedule(optimizer, d_model=d_model, warmup_steps=warmup)

    lrs = []
    for step in range(total_steps):
        # Proper order: optimizer.step() then scheduler.step()
        optimizer.step()
        lrs.append(sched.get_last_lr()[0])
        sched.step()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(total_steps), lrs, label=f"Noam (d_model={d_model}, warmup={warmup})")
    plt.axvline(warmup, color="gray", linestyle="--", label=f"warmup={warmup}")
    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.yscale("log")
    plt.title("Noam Schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


def plot_cosine_restarts(T_0: int, T_mult: int, eta_min: float, total_steps: int, out_path: str):
    optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )

    lrs = []
    for step in range(total_steps):
        optimizer.step()
        lrs.append(sched.get_last_lr()[0])
        sched.step()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(total_steps), lrs, label=f"CosineRestarts (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
    # Mark restart boundaries
    period = T_0
    next_restart = T_0
    while next_restart < total_steps:
        plt.axvline(next_restart, color="gray", linestyle="--", alpha=0.5)
        period = int(period * 2)  # T_mult=2 default
        next_restart += period

    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.yscale("log")
    plt.title("Cosine Annealing with Warm Restarts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler", choices=["noam", "cosine_restarts"], default="noam")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--t0", type=int, default=1000)
    parser.add_argument("--t-mult", type=int, default=2)
    parser.add_argument("--eta-min", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)

    if args.scheduler == "noam":
        out = os.path.join(ROOT, "plots", "lr_noam.png")
        plot_noam(args.d_model, args.warmup, args.steps, out)
    else:
        out = os.path.join(ROOT, "plots", "lr_cosine_restarts.png")
        plot_cosine_restarts(args.t0, args.t_mult, args.eta_min, args.steps, out)


if __name__ == "__main__":
    main()