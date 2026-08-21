"""
Scaling curve experiment (Chinchilla-style): loss vs model size.

Trains four tiny GPTs of growing size on the same Shakespeare data with the
same recipe, then plots the final validation loss against parameter count —
the classic "bigger models learn better" power-law picture, at toy scale.

  python scripts/scale.py --epochs 15
  python scripts/scale.py --epochs 20 --seed 0

Plot: plots/scaling_curve.png
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from transformer.gpt import GPT, CharTokenizer, train  # noqa: E402
from transformer.model import create_look_ahead_mask  # noqa: E402


CONFIGS = [
    {"d_model": 32, "num_heads": 2, "d_ff": 128, "num_layers": 2, "tag": "tiny"},
    {"d_model": 64, "num_heads": 4, "d_ff": 256, "num_layers": 3, "tag": "small"},
    {"d_model": 128, "num_heads": 4, "d_ff": 512, "num_layers": 4, "tag": "base"},
    {"d_model": 256, "num_heads": 8, "d_ff": 1024, "num_layers": 6, "tag": "large"},
]


def eval_val(model: GPT, tokenizer, val_data: torch.Tensor, block_size: int,
             batch_size: int, device) -> float:
    model.eval()
    n = len(val_data) // block_size
    flat = val_data[: n * block_size]           # n*block_size contiguous tokens
    # True next-token targets: predict flat[i+1] from flat[i]. The final
    # sequence's last token has no successor → drop one sequence.
    xs = flat[: (n - 1) * block_size].view(n - 1, block_size)
    ys = flat[1:(n - 1) * block_size + 1].view(n - 1, block_size)
    mask = create_look_ahead_mask(block_size).to(device)
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            x = xs[i:i + batch_size].to(device)
            y = ys[i:i + batch_size].to(device)
            logits = model(x, mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
            count += 1
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-pairs", type=int, default=8000,
                        help="training slices per epoch (default 8000; the real recipe uses 20000)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    with open(os.path.join(ROOT, "data", "shakespeare.txt"), "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    train_data, val_data = data[:-n_val], data[-n_val:]

    print(f"{'tag':<6} {'params':>9} {'val loss':>9} {'ppl':>7}")
    rows = []
    for cfg in CONFIGS:
        tag = cfg["tag"]
        print(f"training {tag}: d_model={cfg['d_model']} layers={cfg['num_layers']} ...")
        model = train(
            tokenizer, train_data, val_data,
            epochs=args.epochs, block_size=args.block_size,
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"], num_layers=cfg["num_layers"],
            batch_size=args.batch_size, save=False, rope=True,
            seed=args.seed, num_pairs=args.num_pairs,
            num_val_pairs=args.num_pairs // 10,
        )
        val_loss = eval_val(model, tokenizer, val_data, args.block_size,
                            args.batch_size, device)
        params = model.count_params()
        rows.append((tag, params, val_loss))
        print(f"{tag:<6} {params:>9,} {val_loss:>9.4f} {math.exp(val_loss):>7.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = [r[0] for r in rows]
        params = [r[1] for r in rows]
        losses = [r[2] for r in rows]
        plt.figure(figsize=(7, 5))
        plt.plot(params, losses, "o-", color="#C44E52")
        for tag, p, loss in zip(tags, params, losses):
            plt.annotate(f"{tag}\n{p / 1000:.0f}k", (p, loss), textcoords="offset points",
                         xytext=(8, 8), fontsize=9)
        plt.xlabel("parameters")
        plt.ylabel("validation loss")
        plt.title("Scaling curve: bigger models learn better")
        plt.xscale("log")
        plt.tight_layout()
        os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)
        out = os.path.join(ROOT, "plots", "scaling_curve.png")
        plt.savefig(out, dpi=120)
        print(f"\nplot saved to {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()