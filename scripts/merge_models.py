#!/usr/bin/env python3
"""
Model merging utilities for mini-GPT.

Supports:
- Linear merging (weighted average)
- SLERP (spherical linear interpolation)
- TIES merging (trim, elect sign, merge)
- DARE merging (delta, rescale, merge)
- LoRA merging (merge LoRA adapters into base model)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transformer.gpt import CharTokenizer, GPT
from transformer.bpe import BPETokenizer

# Register tokenizer classes for unpickling
import __main__
__main__.CharTokenizer = CharTokenizer
__main__.BPETokenizer = BPETokenizer


def load_model(ckpt_path, device="cpu"):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Extract config from checkpoint (handle different formats)
    vocab_size = ckpt.get("vocab_size")
    max_len = ckpt.get("max_len")
    rope = ckpt.get("rope", True)
    num_kv_heads = ckpt.get("num_kv_heads", None)
    
    # If not in checkpoint, try to infer from tokenizer
    if vocab_size is None and "tokenizer" in ckpt:
        tokenizer = ckpt["tokenizer"]
        vocab_size = tokenizer.vocab_size
        max_len = getattr(tokenizer, 'max_len', 128)
    
    # Defaults
    if vocab_size is None:
        vocab_size = 65
    if max_len is None:
        max_len = 128
    
    model = GPT(
        vocab_size=vocab_size,
        max_len=max_len,
        rope=rope,
        num_kv_heads=num_kv_heads
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    return model, ckpt


def linear_merge(models, weights):
    """Linear weighted average of models."""
    assert len(models) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    
    merged_state = {}
    for key in models[0].keys():
        merged_state[key] = sum(w * m[key] for w, m in zip(weights, models))
    
    return merged_state


def slerp_merge(models, weights, t=0.5):
    """SLERP merge (spherical linear interpolation) for two models."""
    assert len(models) == 2
    assert len(weights) == 2
    
    merged_state = {}
    for key in models[0].keys():
        w1, w2 = models[0][key], models[1][key]
        
        # Flatten
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()
        
        # Compute angle
        dot = (w1_flat * w2_flat).sum()
        norm1 = w1_flat.norm()
        norm2 = w2_flat.norm()
        cos_theta = dot / (norm1 * norm2 + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        
        # SLERP
        sin_theta = torch.sin(theta)
        w_merged = (torch.sin((1 - t) * theta) / sin_theta) * w1 + \
                   (torch.sin(t * theta) / sin_theta) * w2
        
        merged_state[key] = w_merged.reshape(w1.shape)
    
    return merged_state


def ties_merge(models, weights, density=0.5):
    """TIES merging: Trim, Elect Sign, Merge."""
    assert len(models) == len(weights)
    
    # Compute task vectors (difference from base)
    # Assuming first model is base
    base = models[0]
    task_vectors = []
    for i in range(1, len(models)):
        tv = {}
        for key in base.keys():
            tv[key] = models[i][key] - base[key]
        task_vectors.append(tv)
    
    merged_state = {}
    for key in base.keys():
        # Collect all task vectors for this key
        tvs = [tv[key] for tv in task_vectors]
        
        # TRIM: Keep only top-k% by magnitude
        stacked = torch.stack(tvs)
        k = int(density * stacked.numel() / len(tvs))
        # Find threshold for top-k
        flat = stacked.abs().flatten()
        threshold = flat.kthvalue(max(1, flat.numel() - k)).values
        
        # ELECT SIGN: Determine majority sign
        signs = torch.sign(stacked)
        majority_sign = torch.sign(signs.sum(dim=0))
        
        # TRIM: Zero out values below threshold
        stacked = stacked * (stacked.abs() >= threshold).float()
        tvs = stacked.unbind(0)
        
        # Zero out minority signs
        for i in range(len(tvs)):
            tvs[i] = tvs[i] * (signs[i] == majority_sign).float()
        
        # MERGE: Weighted average of trimmed vectors
        merged_tv = sum(w * tv for w, tv in zip(weights[1:], tvs))
        
        merged_state[key] = base[key] + merged_tv
    
    return merged_state


def dare_merge(models, weights, density=0.5, rescale=1.0):
    """DARE merging: Drop And REscale."""
    assert len(models) == len(weights)
    
    base = models[0]
    task_vectors = []
    for i in range(1, len(models)):
        tv = {}
        for key in base.keys():
            tv[key] = models[i][key] - base[key]
        task_vectors.append(tv)
    
    merged_state = {}
    for key in base.keys():
        tvs = [tv[key] for tv in task_vectors]
        
        # DROP: Randomly zero out weights
        merged_tv = torch.zeros_like(base[key])
        for i, tv in enumerate(tvs):
            mask = torch.rand_like(tv) < density
            merged_tv += weights[i + 1] * tv * mask
        
        # RESCALE
        merged_tv = merged_tv * (rescale / density)
        
        merged_state[key] = base[key] + merged_tv
    
    return merged_state


def lora_merge(base_model, lora_ckpt_path, scaling=1.0, device="cpu"):
    """Merge LoRA adapter into base model."""
    # Load LoRA checkpoint
    lora_ckpt = torch.load(lora_ckpt_path, map_location=device, weights_only=False)
    
    # LoRA weights are stored as lora_A and lora_B for each layer
    base_state = base_model.state_dict()
    merged_state = base_state.copy()
    
    for key, value in lora_ckpt.get("model", {}).items():
        if "lora_A" in key:
            # Find corresponding lora_B
            layer_name = key.replace("lora_A", "")
            lora_b_key = layer_name + "lora_B"
            
            if lora_b_key in lora_ckpt["model"]:
                lora_A = value
                lora_B = lora_ckpt["model"][lora_b_key]
                
                # Compute LoRA delta: B @ A * scaling
                delta = (lora_B @ lora_A) * scaling
                
                # Find base weight key
                base_key = layer_name.replace("lora_", "").rstrip(".")
                if base_key in base_state:
                    merged_state[base_key] = base_state[base_key] + delta.to(base_state[base_key].device)
    
    return merged_state


def save_merged_model(merged_state, template_ckpt, output_path, quantization=None):
    """Save merged model."""
    save_dict = {
        "model": merged_state,
        "vocab_size": template_ckpt.get("vocab_size", 65),
        "max_len": template_ckpt.get("max_len", 128),
        "rope": template_ckpt.get("rope", True),
        "num_kv_heads": template_ckpt.get("num_kv_heads", None),
    }
    if quantization:
        save_dict["quantization"] = quantization
    
    torch.save(save_dict, output_path)
    print(f"Saved merged model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge mini-GPT models")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Model checkpoint paths")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="Merge weights (sum to 1)")
    parser.add_argument("--method", type=str, default="linear",
                        choices=["linear", "slerp", "ties", "dare", "lora"],
                        help="Merging method")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--density", type=float, default=0.5, help="Density for TIES/DARE")
    parser.add_argument("--rescale", type=float, default=1.0, help="Rescale factor for DARE")
    parser.add_argument("--slerp-t", type=float, default=0.5, help="SLERP interpolation parameter")
    parser.add_argument("--lora-scaling", type=float, default=1.0, help="LoRA scaling factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    args = parser.parse_args()
    
    if args.weights is None:
        args.weights = [1.0 / len(args.models)] * len(args.models)
    
    assert len(args.models) == len(args.weights), "Number of models must match number of weights"
    assert abs(sum(args.weights) - 1.0) < 1e-6, "Weights must sum to 1"
    
    print(f"Loading {len(args.models)} models...")
    models = []
    ckpts = []
    for path in args.models:
        model, ckpt = load_model(path, args.device)
        models.append(model.state_dict())
        ckpts.append(ckpt)
        print(f"  Loaded {path}")
    
    print(f"\nMerging with {args.method}...")
    
    if args.method == "linear":
        merged_state = linear_merge(models, args.weights)
    elif args.method == "slerp":
        if len(models) != 2:
            raise ValueError("SLERP only supports 2 models")
        merged_state = slerp_merge(models, args.weights, t=args.slerp_t)
    elif args.method == "ties":
        merged_state = ties_merge(models, args.weights, density=args.density)
    elif args.method == "dare":
        merged_state = dare_merge(models, args.weights, density=args.density, rescale=args.rescale)
    elif args.method == "lora":
        if len(models) != 2:
            raise ValueError("LoRA merge requires base model + 1 LoRA checkpoint")
        base_model, _ = load_model(args.models[0], args.device)
        merged_state = lora_merge(base_model, args.models[1], scaling=args.lora_scaling, device=args.device)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    save_merged_model(merged_state, ckpts[0], args.output)
    
    # Verify
    print("\nVerifying merged model...")
    verify_model, _ = load_model(args.output, args.device)
    test_input = torch.randint(0, 65, (1, 128), device=args.device)
    
    with torch.no_grad():
        out = verify_model(test_input)
    print(f"  Output shape: {out.shape}")
    print("  Merge successful!")


if __name__ == "__main__":
    main()