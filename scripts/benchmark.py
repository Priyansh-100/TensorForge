"""
Training throughput benchmark: fp32 vs AMP vs torch.compile vs AMP+compile.

Trains the SAME tiny GPT for the same number of steps under each configuration
and reports tokens/sec. Speedups here are qualitative — the point is to show
how much free speed each knob buys on your hardware:

  python scripts/benchmark.py                 # all four configs
  python scripts/benchmark.py --only amp,compile
  python scripts/benchmark.py --steps 50 --batch-size 64

Plot: plots/benchmark.png (tok/s bar chart)
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from transformer.gpt import GPT  # noqa: E402
from transformer.model import create_look_ahead_mask  # noqa: E402


def make_data(seed: int, steps: int, batch_size: int, block_size: int, vocab: int, device):
    torch.manual_seed(seed)
    return torch.randint(2, vocab, (steps * batch_size * block_size,), device=device)


def bench_config(device, label: str, amp: bool, comp: bool, data: torch.Tensor,
                 block_size: int, batch_size: int, steps: int, vocab: int):
    torch.manual_seed(0)
    model = GPT(vocab_size=vocab, d_model=128, num_heads=4, d_ff=512,
                num_layers=4, max_len=block_size, rope=True).to(device)

    if comp:
        try:
            model = torch.compile(model)
        except Exception as e:
            return label, None, f"torch.compile unavailable: {e}"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    mask = create_look_ahead_mask(block_size).to(device)

    use_amp = amp and device.type in ("cuda", "mps")
    if amp and not use_amp:
        return label, None, "AMP n/a on CPU"

    # warmup (also triggers torch.compile's one-time compilation)
    x = data[: batch_size * block_size].view(batch_size, block_size)
    y = x.roll(-1, dims=1)
    with torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext():
        loss = criterion(model(x, mask).view(-1, vocab), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    t0 = time.time()
    for s in range(steps):
        x = data[s * batch_size * block_size:(s + 1) * batch_size * block_size].view(batch_size, block_size)
        y = x.roll(-1, dims=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext():
            loss = criterion(model(x, mask).view(-1, vocab), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    dt = time.time() - t0
    toks = steps * batch_size * block_size
    return label, toks / dt, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--only", type=str, default="fp32,amp,compile,amp+compile",
                        help="comma-separated subset of the four configs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}\n")

    configs = {
        "fp32": dict(amp=False, comp=False),
        "amp": dict(amp=True, comp=False),
        "compile": dict(amp=False, comp=True),
        "amp+compile": dict(amp=True, comp=True),
    }
    only = [c.strip() for c in args.only.split(",") if c.strip()]

    data = make_data(0, args.steps, args.batch_size, args.block_size, vocab=65, device=device)
    results = []
    for label in only:
        if label not in configs:
            print(f"  unknown config '{label}' — use one of: {', '.join(configs)}")
            continue
        print(f"running {label} ...")
        kw = configs[label]
        res = bench_config(device, label, kw["amp"], kw["comp"], data,
                           args.block_size, args.batch_size, args.steps, vocab=65)
        results.append(res)
        if res[2]:
            print(f"  {label}: {res[2]}")
        else:
            print(f"  {label}: {res[1]:,.0f} tok/s")

    print("\n{:<13} {:>14} {:>12}".format("config", "tok/s", "speedup"))
    base = next((r for r in results if r[0] == "fp32" and r[1]), None)
    for label, rate, err in results:
        if rate is None:
            print(f"{label:<13} {'n/a':>14} {'—':>12}  ({err})")
            continue
        speedup = rate / base[1] if base else 1.0
        print(f"{label:<13} {rate:>14,.0f} {speedup:>12.2f}x")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r[0] for r in results]
        rates = [r[1] for r in results]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, [r or 0 for r in rates], color="#4C72B0")
        for i, r in enumerate(rates):
            if r:
                plt.text(i, r, f"{r:,.0f}", ha="center", va="bottom", fontsize=9)
        plt.ylabel("training tokens/sec")
        plt.title(f"Training throughput on {device}")
        plt.tight_layout()
        os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
        out = os.path.join(ROOT, "plots", "benchmark.png")
        plt.savefig(out, dpi=120)
        print(f"\nplot saved to {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()