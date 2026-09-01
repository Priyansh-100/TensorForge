#!/usr/bin/env python3
"""
Eval Harness for mini-GPT.

Automatically evaluates all checkpoints and generates a comparison table
for the README (Part 13). Runs validation, perplexity, generation, and
benchmarking for each checkpoint.

Usage:
  python scripts/eval_harness.py                    # eval all checkpoints
  python scripts/eval_harness.py --ckpt ckpt.pt    # eval single checkpoint
  python scripts/eval_harness.py --bench            # run throughput benchmark
  python scripts/eval_harness.py --output results.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer.gpt import GPT, CharTokenizer
from transformer.model import create_look_ahead_mask, NoamSchedule


CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt")


class CharDataset:
    def __init__(self, data, block_size, num_pairs):
        self.data = data
        self.block_size = block_size
        self.num_pairs = num_pairs

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, _):
        hi = len(self.data) - self.block_size - 1
        idx = torch.randint(0, max(hi, 1), ())
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    tokenizer = ckpt["tokenizer"]
    has_learned_pos = "pos_embedding.weight" in ckpt["model"]
    
    # Infer architecture from checkpoint
    state_dict = ckpt["model"]
    d_model = state_dict["token_embedding.weight"].shape[1]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    num_layers = sum(1 for k in state_dict.keys() if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    
    # Infer num_heads from W_q shape: [num_heads * d_k, d_model]
    w_q_shape = state_dict["blocks.0.self_attn.W_q.weight"].shape
    # W_q is [out_features, in_features] = [num_heads * d_k, d_model] or [d_model, d_model]
    in_dim = w_q_shape[1]
    out_dim = w_q_shape[0]
    if out_dim == in_dim:
        # Standard MHA: W_q is [d_model, d_model]
        d_k = d_model // state_dict["blocks.0.self_attn.W_q.weight"].shape[0]
        num_heads = out_dim // d_k
    else:
        d_k = state_dict["blocks.0.self_attn.W_q.weight"].shape[1]
        num_heads = state_dict["blocks.0.self_attn.W_q.weight"].shape[0] // d_k
    
    # Infer num_kv_heads from W_k shape: [num_kv_heads * d_k, d_model]
    w_k_shape = state_dict["blocks.0.self_attn.W_k.weight"].shape
    d_k = state_dict["blocks.0.self_attn.W_q.weight"].shape[1]
    num_kv_heads = w_k_shape[0] // d_k if "num_kv_heads" not in ckpt else ckpt.get("num_kv_heads")
    
    # Verify num_heads is multiple of num_kv_heads
    if num_heads % num_kv_heads != 0:
        # Fallback: try to infer from checkpoint metadata
        num_kv_heads = ckpt.get("num_kv_heads", 1)
        if num_heads % num_kv_heads != 0:
            num_kv_heads = 1  # fallback to MQA
    
    # Infer d_ff from FFN
    d_ff = state_dict["blocks.0.ffn.linear1.weight"].shape[0]

    # Count layers
    num_layers = sum(1 for k in state_dict.keys() if k.startswith("blocks.") and k.endswith(".norm1.weight"))

    model = GPT(
        vocab_size=state_dict["token_embedding.weight"].shape[0],
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=128,
        rope=not has_learned_pos,
        num_kv_heads=num_kv_heads,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, ckpt.get("val_loss")


def evaluate_ppl(model, tokenizer, val_text, block=128, batch_size=32, device="cpu"):
    model.eval()
    ids = tokenizer.encode(val_text)
    total, count = 0.0, 0

    with torch.no_grad():
        for i in range(0, len(ids) - block - 1, block):
            x = torch.tensor([ids[i:i+block]], dtype=torch.long, device=device)
            y = torch.tensor([ids[i+1:i+block+1]], dtype=torch.long, device=device)
            mask = torch.tril(torch.ones(block, block, device=device)).unsqueeze(0).unsqueeze(0)

            logits = model(x, mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
            count += 1

    return math.exp(total / count) if count else float("inf")


def generate_sample(model, tokenizer, prompt, n_tokens=50, device="cpu", temp=0.8, top_p=1.0):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, n_tokens, temperature=0.8, top_p=0.9)
    return tokenizer.decode(out[0].tolist())


def benchmark_throughput(model, tokenizer, block_size=128, batch_size=32, steps=50, device="cpu"):
    model.eval()
    with open(DATA_PATH, "r") as f:
        text = f.read()
    data = torch.tensor(CharTokenizer(text).encode(text), dtype=torch.long)
    train_loader = DataLoader(CharDataset(data, block_size, 20000), batch_size=32, shuffle=True)

    model.eval()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    mask = torch.tril(torch.ones(128, 128, device=device)).unsqueeze(0).unsqueeze(0)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, mask)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        with torch.no_grad():
            _ = model(x, mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    tok_s = steps * batch_size * 128 / dt
    return tok_s


def load_checkpoints(ckpt_dir):
    """Find all .pt checkpoints in directory."""
    checkpoints = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt"):
            checkpoints.append(os.path.join(ckpt_dir, f))
    return sorted(checkpoints)


def eval_checkpoint(path, val_text, device="cpu"):
    """Evaluate a single checkpoint."""
    print(f"\n=== Evaluating {os.path.basename(path)} ===")

    model, tokenizer, stored_val = load_checkpoint(path)
    model.to(device)

    results = {
        "checkpoint": os.path.basename(path),
        "stored_val_loss": stored_val,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Validation perplexity
    ppl = evaluate_ppl(model, tokenizer, val_text, device=device)
    results["val_ppl"] = ppl

    # Generation sample
    sample = generate_sample(model, tokenizer, "To be, or not to be", 50, device)
    results["sample"] = sample[:100]

    # Throughput
    tok_s = benchmark_throughput(model, tokenizer, device=device)
    results["tok_s"] = tok_s

    print(f"  Params: {results['params']:,}")
    print(f"  Stored val loss: {results['stored_val_loss']:.4f}")
    print(f"  Val ppl: {results['val_ppl']:.2f}")
    print(f"  Throughput: {results['tok_s']:,.0f} tok/s")
    print(f"  Sample: {results['sample']}...")

    return results


def load_checkpoints(ckpt_dir):
    """Find all .pt checkpoints in directory."""
    checkpoints = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt"):
            checkpoints.append(os.path.join(ckpt_dir, f))
    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Single checkpoint to evaluate")
    parser.add_argument("--bench", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    with open(DATA_PATH, "r") as f:
        text = f.read()
    val_text = text[-20000:]

    if args.ckpt:
        checkpoints = [args.ckpt]
    else:
        checkpoints = load_checkpoints(CKPT_DIR)

    all_results = []
    for ckpt in checkpoints:
        try:
            results = eval_checkpoint(ckpt, val_text, device)
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}")
            all_results.append({"checkpoint": os.path.basename(ckpt), "error": str(e)})

    # Print summary table
    print("\n\n=== EVALUATION SUMMARY ===")
    print(f"{'Checkpoint':<30} {'Params':>10} {'Val Loss':>10} {'PPL':>8} {'Tok/s':>10}")
    print("-" * 70)
    for r in all_results:
        if "error" not in r:
            print(f"{r['checkpoint']:<30} {r['params']:>10,} {r['stored_val_loss']:>10.4f} {r['val_ppl']:>8.2f} {r['tok_s']:>10,}")
        else:
            print(f"{r['checkpoint']:<30} ERROR: {r['error']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()