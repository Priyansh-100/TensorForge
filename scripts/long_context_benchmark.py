#!/usr/bin/env python3
"""
Long-context benchmark: evaluate generation quality vs context length.

Tests how model perplexity and generation quality degrade as context
exceeds the 128-token training window, with and without RoPE scaling.

Usage:
  python scripts/long_context_benchmark.py --ckpt checkpoints/gpt_rope.pt
  python scripts/long_context_benchmark.py --ckpt checkpoints/gpt_rope.pt --rope-scaling ntk --rope-scaling-factor 2.0
  python scripts/long_context_benchmark.py --ckpt checkpoints/gpt_rope.pt --context-lengths 128,256,512,1024 --n-samples 10

Output: markdown table + plots/long_context_benchmark.png
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from transformer.gpt import GPT
from transformer.model import create_look_ahead_mask

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt")


def load_checkpoint(path: str, rope_scaling: str = "none", rope_scaling_factor: float = 1.0):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = ckpt["tokenizer"]
    num_kv = ckpt.get("num_kv_heads")
    has_learned_pos = "pos_embedding.weight" in ckpt["model"]
    
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        max_len=128,
        rope=not has_learned_pos,
        num_kv_heads=num_kv,
        rope_scaling=rope_scaling,
        rope_scaling_factor=rope_scaling_factor,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def measure_ppl_at_context(model, tokenizer, val_text, context_len, block_size, device, n_samples=50):
    """Measure perplexity using a fixed context window."""
    model.eval()
    ids = tokenizer.encode(val_text)
    total_loss, count = 0.0, 0
    
    with torch.no_grad():
        for i in range(0, min(n_samples * block_size, len(ids) - context_len - 1), block_size):
            x = torch.tensor([ids[i:i + context_len]], dtype=torch.long, device=device)
            y = torch.tensor([ids[i + 1:i + context_len + 1]], dtype=torch.long, device=device)
            mask = create_look_ahead_mask(context_len).to(device)
            logits = model(x, mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            total_loss += loss.item()
            count += 1
            if count >= n_samples:
                break
    
    return math.exp(total_loss / count) if count > 0 else float("inf")


def generate_sample(model, tokenizer, prompt, n_tokens, device, temperature=0.8):
    """Generate text with temperature sampling."""
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate_cached(idx, n_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT_DIR, "gpt_rope.pt"))
    parser.add_argument("--context-lengths", type=str, default="128,256,512,1024",
                        help="comma-separated context lengths to test")
    parser.add_argument("--rope-scaling", choices=["none", "linear", "ntk"], default="none")
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=50, help="samples per context length for ppl")
    parser.add_argument("--n-tokens", type=int, default=200, help="tokens to generate for quality check")
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    args = parser.parse_args()

    context_lengths = [int(x) for x in args.context_lengths.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Context lengths: {context_lengths}")
    print(f"RoPE scaling: {args.rope_scaling} x{args.rope_scaling_factor}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    val_text = text[-20000:]  # last 20k chars for validation

    model, tokenizer = load_checkpoint(args.ckpt, args.rope_scaling, args.rope_scaling_factor)
    model.to(device)

    print(f"\nModel vocab: {tokenizer.vocab_size} | params: {model.count_params():,}")

    results = []
    for ctx in context_lengths:
        print(f"\n=== Context length: {ctx} ===")
        ppl = measure_ppl_at_context(model, tokenizer, val_text, ctx, 128, device, args.n_samples)
        print(f"  Perplexity: {ppl:.2f}")

        # Generate a sample
        sample_text = generate_sample(model, tokenizer, args.prompt, min(args.n_tokens, ctx), device)
        print(f"  Sample: {sample_text[:100]}...")

        results.append({
            "context_length": ctx,
            "perplexity": ppl,
            "sample": sample_text[:200]
        })

    # Print markdown table
    print("\n\n## Long-Context Benchmark Results\n")
    print("| context length | perplexity | sample preview |")
    print("|---:|---:|---|")
    for r in results:
        print(f"| {r['context_length']} | {r['perplexity']:.2f} | {r['sample'][:80]}... |")

    # Save JSON
    out_json = {
        "checkpoint": os.path.basename(args.ckpt),
        "rope_scaling": args.rope_scaling,
        "rope_scaling_factor": args.rope_scaling_factor,
        "results": results
    }
    with open("long_context_benchmark.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print("\nSaved long_context_benchmark.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ctxs = [r["context_length"] for r in results]
        ppls = [r["perplexity"] for r in results]

        plt.figure(figsize=(7, 5))
        plt.plot(ctxs, ppls, "o-", color="#C44E52")
        for ctx, ppl in zip(ctxs, ppls):
            plt.annotate(f"{ppl:.1f}", (ctx, ppl), textcoords="offset points", xytext=(8, 8), fontsize=9)
        plt.xlabel("Context length")
        plt.ylabel("Perplexity")
        plt.title(f"Long-context quality ({args.rope_scaling} x{args.rope_scaling_factor})")
        plt.xscale("log", base=2)
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        out_png = os.path.join("plots", "long_context_benchmark.png")
        plt.savefig(out_png, dpi=120)
        print(f"\nPlot saved to {out_png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()