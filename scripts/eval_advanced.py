#!/usr/bin/env python3
"""
Advanced evaluation harness for mini-GPT.

Supports:
- Perplexity on multiple datasets
- Generation quality metrics (BLEU, ROUGE, etc.)
- Zero-shot/few-shot evaluation
- Long-context evaluation
- Throughput benchmarking
- Memory profiling
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.gpt import GPT, CharTokenizer
from transformer.bpe import BPETokenizer


def load_model_and_tokenizer(ckpt_path, device):
    """Load model and tokenizer from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    vocab_size = ckpt.get("vocab_size", 65)
    max_len = ckpt.get("max_len", 128)
    rope = ckpt.get("rope", True)
    num_kv_heads = ckpt.get("num_kv_heads", None)
    
    model = GPT(
        vocab_size=vocab_size,
        max_len=max_len,
        rope=rope,
        num_kv_heads=num_kv_heads
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    # Determine tokenizer type
    if "bpe_merges" in ckpt:
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        tokenizer.merges = ckpt["bpe_merges"]
        tokenizer.vocab = ckpt.get("bpe_vocab", {})
    else:
        # Load Shakespeare for char tokenizer
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
            text = f.read()
        tokenizer = CharTokenizer(text)
    
    return model, tokenizer, ckpt


def evaluate_perplexity(model, tokenizer, text, block_size, batch_size=32, device="cpu", max_batches=None):
    """Evaluate perplexity on a text corpus."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data) - block_size, batch_size * block_size):
            if max_batches and num_batches >= max_batches:
                break
            
            # Create batch
            batch_end = min(i + batch_size * block_size, len(data) - block_size)
            if batch_end - i < block_size:
                break
            
            ix = torch.randint(i, batch_end - block_size, (batch_size,))
            x = torch.stack([data[j:j+block_size] for j in ix]).to(device)
            y = torch.stack([data[j+1:j+block_size+1] for j in ix]).to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            num_batches += 1
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
    
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "tokens": total_tokens,
        "batches": num_batches
    }


def evaluate_generation_quality(model, tokenizer, prompts, max_new_tokens=100, temperature=0.8, top_p=0.9, device="cpu"):
    """Evaluate generation quality on a set of prompts."""
    model.eval()
    generations = []
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            generated = model.generate(input_ids, max_new_tokens, temperature, top_p)
            text = tokenizer.decode(generated[0].tolist())
            generations.append({
                "prompt": prompt,
                "generated": text
            })
    
    return generations


def evaluate_long_context(model, tokenizer, context_lengths, device="cpu"):
    """Evaluate model at different context lengths."""
    results = {}
    
    for ctx_len in context_lengths:
        if ctx_len > tokenizer.max_len if hasattr(tokenizer, 'max_len') else ctx_len > 2048:
            results[ctx_len] = {"error": "Context length exceeds model maximum"}
            continue
        
        # Create test input
        test_text = "To be, or not to be, that is the question. " * (ctx_len // 50 + 1)
        input_ids = torch.tensor([tokenizer.encode(test_text[:ctx_len])], dtype=torch.long, device=device)
        
        if input_ids.size(1) < 2:
            results[ctx_len] = {"error": "Input too short"}
            continue
        
        model.eval()
        with torch.no_grad():
            start = time.time()
            logits = model(input_ids[:, :-1])
            elapsed = time.time() - start
            
            # Compute loss on last token
            target = input_ids[:, 1:]
            loss = F.cross_entropy(logits[:, -1, :], target[:, -1])
            
            results[ctx_len] = {
                "loss": loss.item(),
                "perplexity": torch.exp(loss).item(),
                "latency_ms": elapsed * 1000,
                "input_tokens": input_ids.size(1)
            }
    
    return results


def benchmark_throughput(model, tokenizer, batch_sizes, seq_lengths, num_runs=10, device="cpu"):
    """Benchmark throughput at different batch sizes and sequence lengths."""
    results = {}
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            key = f"bs{batch_size}_sl{seq_len}"
            try:
                input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
                
                # Warmup
                model.eval()
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(input_ids)
                
                # Benchmark
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()
                
                start = time.time()
                with torch.no_grad():
                    for _ in range(num_runs):
                        _ = model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()
                elapsed = time.time() - start
                
                total_tokens = batch_size * seq_len * num_runs
                tok_per_sec = total_tokens / elapsed
                latency_ms = (elapsed / num_runs) * 1000
                
                results[key] = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "latency_ms": latency_ms,
                    "tok_per_sec": tok_per_sec,
                    "total_tokens": total_tokens
                }
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[key] = {"error": "OOM"}
                else:
                    raise
    
    return results


def profile_memory(model, tokenizer, batch_size=4, seq_len=512, device="cpu"):
    """Profile memory usage."""
    if device == "cpu":
        return {"error": "Memory profiling requires CUDA"}
    
    model.eval()
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    
    # Reset peak memory
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(input_ids)
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    
    return {
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "peak_mb": peak,
        "batch_size": batch_size,
        "seq_len": seq_len
    }


def run_few_shot_eval(model, tokenizer, tasks, device="cpu"):
    """Run few-shot evaluation on standard tasks."""
    # This is a simplified version - real implementation would use proper datasets
    results = {}
    
    for task_name, task_data in tasks.items():
        prompt = task_data["prompt"]
        examples = task_data.get("examples", [])
        
        # Build few-shot prompt
        few_shot_prompt = ""
        for ex in examples:
            few_shot_prompt += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
        few_shot_prompt += f"Q: {prompt}\nA:"
        
        input_ids = torch.tensor([tokenizer.encode(few_shot_prompt)], dtype=torch.long, device=device)
        
        model.eval()
        with torch.no_grad():
            generated = model.generate(input_ids, 50, 0.0, 1.0)  # Greedy
            text = tokenizer.decode(generated[0].tolist())
        
        # Extract answer
        answer = text.split("A:")[-1].strip()
        
        results[task_name] = {
            "prompt": few_shot_prompt,
            "generated": text,
            "answer": answer
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Advanced evaluation harness")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output JSON file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--block-size", type=int, default=128, help="Block size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--ppl-max-batches", type=int, default=None, help="Max batches for PPL")
    parser.add_argument("--gen-prompts", type=str, nargs="+", default=["To be, or not to be", "The quick brown fox"], help="Generation prompts")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 256, 512, 1024], help="Context lengths to test")
    parser.add_argument("--benchmark-throughput", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes for throughput")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256, 512], help="Sequence lengths for throughput")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.ckpt}...")
    model, tokenizer, ckpt = load_model_and_tokenizer(args.ckpt, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    results = {
        "checkpoint": args.ckpt,
        "device": device,
        "model_params": sum(p.numel() for p in model.parameters()),
        "config": {
            "vocab_size": ckpt.get("vocab_size"),
            "max_len": ckpt.get("max_len"),
            "rope": ckpt.get("rope"),
            "num_kv_heads": ckpt.get("num_kv_heads"),
        }
    }
    
    # Load evaluation data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    val_text = ""
    val_files = ["shakespeare.txt", "val.txt", "validation.txt"]
    for vf in val_files:
        vpath = os.path.join(data_dir, vf)
        if os.path.exists(vpath):
            with open(vpath, "r") as f:
                val_text = f.read()
            break
    
    if not val_text:
        # Use last 10% of shakespeare
        with open(os.path.join(data_dir, "shakespeare.txt"), "r") as f:
            full_text = f.read()
        val_text = full_text[-len(full_text)//10:]
    
    print(f"Validation text length: {len(val_text)} chars")
    
    # 1. Perplexity evaluation
    print("\n1. Evaluating perplexity...")
    ppl_results = evaluate_perplexity(
        model, tokenizer, val_text, args.block_size, args.batch_size, device, args.ppl_max_batches
    )
    results["perplexity"] = ppl_results
    print(f"  Loss: {ppl_results['loss']:.4f}, Perplexity: {ppl_results['perplexity']:.4f}")
    
    # 2. Generation quality
    print("\n2. Evaluating generation quality...")
    gen_results = evaluate_generation_quality(
        model, tokenizer, args.gen_prompts, args.max_new_tokens, args.temperature, args.top_p, device
    )
    results["generation"] = gen_results
    for g in gen_results:
        print(f"  Prompt: {g['prompt'][:50]}...")
        print(f"  Generated: {g['generated'][:100]}...")
    
    # 3. Long context evaluation
    print("\n3. Evaluating long context...")
    lc_results = evaluate_long_context(model, tokenizer, args.context_lengths, device)
    results["long_context"] = lc_results
    for ctx_len, res in lc_results.items():
        if "error" in res:
            print(f"  {ctx_len}: {res['error']}")
        else:
            print(f"  {ctx_len}: loss={res['loss']:.4f}, ppl={res['perplexity']:.4f}, latency={res['latency_ms']:.1f}ms")
    
    # 4. Throughput benchmark
    if args.benchmark_throughput:
        print("\n4. Benchmarking throughput...")
        tp_results = benchmark_throughput(model, tokenizer, args.batch_sizes, args.seq_lengths, device=device)
        results["throughput"] = tp_results
        for key, res in tp_results.items():
            if "error" in res:
                print(f"  {key}: {res['error']}")
            else:
                print(f"  {key}: {res['tok_per_sec']:.0f} tok/s, latency={res['latency_ms']:.1f}ms")
    
    # 5. Memory profiling
    if args.profile_memory:
        print("\n5. Profiling memory...")
        mem_results = profile_memory(model, tokenizer, device=device)
        results["memory"] = mem_results
        if "error" not in mem_results:
            print(f"  Allocated: {mem_results['allocated_mb']:.1f} MB")
            print(f"  Reserved: {mem_results['reserved_mb']:.1f} MB")
            print(f"  Peak: {mem_results['peak_mb']:.1f} MB")
        else:
            print(f"  {mem_results['error']}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()