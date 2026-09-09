#!/usr/bin/env python3
"""
Quantization utilities for mini-GPT using torchao (modern PyTorch quantization).

Supports:
- Dynamic INT8 quantization (weights INT8, activations dynamic INT8)
- Weight-only INT8 quantization (weights INT8, activations FP16/FP32)
- Weight-only INT4 quantization (weights INT4, activations FP16/FP32)
- Float8 quantization (experimental)

Requires: pip install torchao
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.gpt import GPT, CharTokenizer


def dynamic_int8_quantize(model):
    """Apply dynamic INT8 quantization (weights + activations INT8)."""
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    except ImportError:
        raise ImportError("torchao required: pip install torchao")
    
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    return model


def weight_only_int8_quantize(model):
    """Apply weight-only INT8 quantization."""
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError:
        raise ImportError("torchao required: pip install torchao")
    
    quantize_(model, Int8WeightOnlyConfig())
    return model


def weight_only_int4_quantize(model, group_size=128):
    """Apply weight-only INT4 quantization."""
    try:
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
    except ImportError:
        raise ImportError("torchao required: pip install torchao")
    
    quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
    return model


def weight_only_int2_quantize(model, group_size=64):
    """Apply weight-only INT2 quantization (experimental)."""
    try:
        from torchao.quantization import Int2WeightOnlyConfig, quantize_
    except ImportError:
        raise ImportError("torchao required: pip install torchao")
    
    # Check if Int2WeightOnlyConfig exists
    if not hasattr(torch, 'int2'):
        # Fallback to Int4 with group_size
        from torchao.quantization import Int4WeightOnlyConfig
        quantize_(model, Int4WeightOnlyConfig(group_size=group_size))
        return model
    
    quantize_(model, Int2WeightOnlyConfig(group_size=group_size))
    return model


def float8_quantize(model):
    """Apply Float8 quantization (requires Hopper GPU)."""
    try:
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
    except ImportError:
        raise ImportError("torchao required: pip install torchao")
    
    quantize_(model, Float8WeightOnlyConfig())
    return model


def measure_model_size(model):
    """Measure model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def benchmark_model(model, input_ids, num_runs=10, warmup=3):
    """Benchmark model inference speed."""
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
    
    # Benchmark
    import time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    end = time.time()
    
    elapsed = (end - start) / num_runs
    tokens = input_ids.numel()
    tok_per_sec = tokens / elapsed
    
    return {
        'latency_ms': elapsed * 1000,
        'tok_per_sec': tok_per_sec
    }


def compare_outputs(model_fp, model_q, input_ids, atol=1e-1):
    """Compare outputs between FP and quantized models."""
    model_fp.eval()
    model_q.eval()
    
    with torch.no_grad():
        out_fp = model_fp(input_ids)
        out_q = model_q(input_ids)
    
    max_diff = (out_fp - out_q).abs().max().item()
    mean_diff = (out_fp - out_q).abs().mean().item()
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'passed': max_diff < atol
    }


def main():
    parser = argparse.ArgumentParser(description="Quantize mini-GPT models using torchao")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--method", type=str, default="weight_only_int8",
                        choices=["dynamic_int8", "weight_only_int8", "weight_only_int4", "weight_only_int2", "float8"],
                        help="Quantization method")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for INT4/INT2")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--compare", action="store_true", help="Compare FP vs quantized outputs")
    parser.add_argument("--block-size", type=int, default=128, help="Block size")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")
    
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
    
    # Load tokenizer and data for calibration
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare.txt"), "r") as f:
        text = f.read()
    
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n_val = int(0.1 * len(data))
    _ = data[-n_val:]  # val_data, used for calibration but not needed here
    
    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # Extract config from checkpoint
    vocab_size = ckpt.get("vocab_size", tokenizer.vocab_size)
    max_len = ckpt.get("max_len", args.block_size)
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
    
    print(f"Original model size: {measure_model_size(model):.2f} MB")
    
    # Test input
    test_input = torch.randint(0, vocab_size, (1, args.block_size), device=device)
    
    # Benchmark original
    orig_bench = None
    if args.benchmark:
        print("\nBenchmarking original model...")
        orig_bench = benchmark_model(model, test_input)
        print(f"  Latency: {orig_bench['latency_ms']:.2f} ms")
        print(f"  Throughput: {orig_bench['tok_per_sec']:.0f} tok/s")
    
    # Clone for quantization
    import copy
    model_fp = copy.deepcopy(model)
    model_q = copy.deepcopy(model)
    
    # Quantize
    print(f"\nApplying {args.method} quantization...")
    
    try:
        if args.method == "dynamic_int8":
            quantized = dynamic_int8_quantize(model_q)
        elif args.method == "weight_only_int8":
            quantized = weight_only_int8_quantize(model_q)
        elif args.method == "weight_only_int4":
            quantized = weight_only_int4_quantize(model_q, group_size=args.group_size)
        elif args.method == "weight_only_int2":
            quantized = weight_only_int2_quantize(model_q, group_size=args.group_size)
        elif args.method == "float8":
            quantized = float8_quantize(model_q)
        else:
            raise ValueError(f"Unknown method: {args.method}")
    except ImportError as e:
        print(f"Error: {e}")
        print("Install torchao: pip install torchao")
        return
    except Exception as e:
        print(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    quantized.to(device)
    quantized.eval()
    
    print(f"Quantized model size: {measure_model_size(quantized):.2f} MB")
    print(f"Size reduction: {measure_model_size(model_fp) / measure_model_size(quantized):.2f}x")
    
    # Compare outputs
    if args.compare:
        print("\nComparing outputs...")
        cmp = compare_outputs(model_fp, quantized, test_input)
        print(f"  Max diff: {cmp['max_diff']:.6f}")
        print(f"  Mean diff: {cmp['mean_diff']:.6f}")
        print(f"  Passed (atol=1e-1): {cmp['passed']}")
    
    # Benchmark quantized
    if args.benchmark:
        print("\nBenchmarking quantized model...")
        quant_bench = benchmark_model(quantized, test_input)
        print(f"  Latency: {quant_bench['latency_ms']:.2f} ms")
        print(f"  Throughput: {quant_bench['tok_per_sec']:.0f} tok/s")
        
        if orig_bench:
            speedup = orig_bench['latency_ms'] / quant_bench['latency_ms']
            print(f"  Speedup: {speedup:.2f}x")
    
    # Save
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(args.ckpt)[0]
        output_path = f"{base}_{args.method}.pt"
    
    torch.save({
        "model": quantized.state_dict(),
        "vocab_size": vocab_size,
        "max_len": max_len,
        "rope": rope,
        "num_kv_heads": num_kv_heads,
        "quantization": args.method,
        "group_size": args.group_size if args.method in ["weight_only_int4", "weight_only_int2"] else None,
    }, output_path)
    print(f"\nSaved quantized model to {output_path}")
    print("Note: To load, re-create model architecture and apply same quantization, then load state_dict")


if __name__ == "__main__":
    main()