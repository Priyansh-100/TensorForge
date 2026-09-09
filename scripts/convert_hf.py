#!/usr/bin/env python3
"""
Convert mini-GPT checkpoints to/from Hugging Face format.

Supports:
- Export to HF format (config.json, model.safetensors)
- Import from HF format
- Push to Hugging Face Hub
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))



def create_hf_config(minigpt_config):
    """Create Hugging Face compatible config."""
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "vocab_size": minigpt_config.get("vocab_size", 50257),
        "n_positions": minigpt_config.get("max_len", 1024),
        "n_embd": minigpt_config.get("d_model", 768),
        "n_layer": minigpt_config.get("num_layers", 12),
        "n_head": minigpt_config.get("num_heads", 12),
        "n_inner": minigpt_config.get("d_ff", 3072),
        "activation_function": "gelu_new",
        "resid_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "use_cache": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
        # Custom fields for mini-GPT features
        "rope": minigpt_config.get("rope", True),
        "num_kv_heads": minigpt_config.get("num_kv_heads"),
        "tie_weights": minigpt_config.get("tie_weights", True),
    }


def convert_state_dict_to_hf(minigpt_state, config):
    """Convert mini-GPT state dict to HF GPT-2 format."""
    hf_state = {}
    
    # Embeddings
    hf_state["transformer.wte.weight"] = minigpt_state["tok_emb.weight"]
    hf_state["transformer.wpe.weight"] = minigpt_state["pos_emb.weight"]
    
    # LM head (tied or separate)
    if "lm_head.weight" in minigpt_state:
        hf_state["lm_head.weight"] = minigpt_state["lm_head.weight"]
    else:
        # Weight tying - use token embeddings
        hf_state["lm_head.weight"] = minigpt_state["tok_emb.weight"]
    
    # Final layer norm
    hf_state["transformer.ln_f.weight"] = minigpt_state["ln_f.weight"]
    hf_state["transformer.ln_f.bias"] = minigpt_state["ln_f.bias"]
    
    # Transformer blocks
    num_layers = config["n_layer"]
    for i in range(num_layers):
        prefix = f"blocks.{i}"
        hf_prefix = f"transformer.h.{i}"
        
        # Attention
        # mini-GPT: attn.c_attn.weight (combined Q,K,V) -> HF: separate q,k,v or combined
        if f"{prefix}.attn.c_attn.weight" in minigpt_state:
            # Combined QKV
            c_attn_w = minigpt_state[f"{prefix}.attn.c_attn.weight"]
            c_attn_b = minigpt_state[f"{prefix}.attn.c_attn.bias"]
            
            # HF GPT-2 uses combined c_attn
            hf_state[f"{hf_prefix}.attn.c_attn.weight"] = c_attn_w
            hf_state[f"{hf_prefix}.attn.c_attn.bias"] = c_attn_b
        
        # Attention output projection
        if f"{prefix}.attn.c_proj.weight" in minigpt_state:
            hf_state[f"{hf_prefix}.attn.c_proj.weight"] = minigpt_state[f"{prefix}.attn.c_proj.weight"]
            hf_state[f"{hf_prefix}.attn.c_proj.bias"] = minigpt_state[f"{prefix}.attn.c_proj.bias"]
        
        # Layer norm 1
        if f"{prefix}.ln_1.weight" in minigpt_state:
            hf_state[f"{hf_prefix}.ln_1.weight"] = minigpt_state[f"{prefix}.ln_1.weight"]
            hf_state[f"{hf_prefix}.ln_1.bias"] = minigpt_state[f"{prefix}.ln_1.bias"]
        
        # MLP
        if f"{prefix}.mlp.c_fc.weight" in minigpt_state:
            hf_state[f"{hf_prefix}.mlp.c_fc.weight"] = minigpt_state[f"{prefix}.mlp.c_fc.weight"]
            hf_state[f"{hf_prefix}.mlp.c_fc.bias"] = minigpt_state[f"{prefix}.mlp.c_fc.bias"]
        
        if f"{prefix}.mlp.c_proj.weight" in minigpt_state:
            hf_state[f"{hf_prefix}.mlp.c_proj.weight"] = minigpt_state[f"{prefix}.mlp.c_proj.weight"]
            hf_state[f"{hf_prefix}.mlp.c_proj.bias"] = minigpt_state[f"{prefix}.mlp.c_proj.bias"]
        
        # Layer norm 2
        if f"{prefix}.ln_2.weight" in minigpt_state:
            hf_state[f"{hf_prefix}.ln_2.weight"] = minigpt_state[f"{prefix}.ln_2.weight"]
            hf_state[f"{hf_prefix}.ln_2.bias"] = minigpt_state[f"{prefix}.ln_2.bias"]
    
    return hf_state


def convert_state_dict_from_hf(hf_state, config):
    """Convert HF GPT-2 state dict to mini-GPT format."""
    minigpt_state = {}
    
    # Embeddings
    minigpt_state["tok_emb.weight"] = hf_state["transformer.wte.weight"]
    minigpt_state["pos_emb.weight"] = hf_state["transformer.wpe.weight"]
    
    # LM head
    if "lm_head.weight" in hf_state:
        minigpt_state["lm_head.weight"] = hf_state["lm_head.weight"]
    else:
        # Weight tied
        minigpt_state["lm_head.weight"] = hf_state["transformer.wte.weight"]
    
    # Final layer norm
    minigpt_state["ln_f.weight"] = hf_state["transformer.ln_f.weight"]
    minigpt_state["ln_f.bias"] = hf_state["transformer.ln_f.bias"]
    
    # Transformer blocks
    num_layers = config["n_layer"]
    for i in range(num_layers):
        prefix = f"blocks.{i}"
        hf_prefix = f"transformer.h.{i}"
        
        # Attention
        minigpt_state[f"{prefix}.attn.c_attn.weight"] = hf_state[f"{hf_prefix}.attn.c_attn.weight"]
        minigpt_state[f"{prefix}.attn.c_attn.bias"] = hf_state[f"{hf_prefix}.attn.c_attn.bias"]
        
        minigpt_state[f"{prefix}.attn.c_proj.weight"] = hf_state[f"{hf_prefix}.attn.c_proj.weight"]
        minigpt_state[f"{prefix}.attn.c_proj.bias"] = hf_state[f"{hf_prefix}.attn.c_proj.bias"]
        
        # Layer norm 1
        minigpt_state[f"{prefix}.ln_1.weight"] = hf_state[f"{hf_prefix}.ln_1.weight"]
        minigpt_state[f"{prefix}.ln_1.bias"] = hf_state[f"{hf_prefix}.ln_1.bias"]
        
        # MLP
        minigpt_state[f"{prefix}.mlp.c_fc.weight"] = hf_state[f"{hf_prefix}.mlp.c_fc.weight"]
        minigpt_state[f"{prefix}.mlp.c_fc.bias"] = hf_state[f"{hf_prefix}.mlp.c_fc.bias"]
        
        minigpt_state[f"{prefix}.mlp.c_proj.weight"] = hf_state[f"{hf_prefix}.mlp.c_proj.weight"]
        minigpt_state[f"{prefix}.mlp.c_proj.bias"] = hf_state[f"{hf_prefix}.mlp.c_proj.bias"]
        
        # Layer norm 2
        minigpt_state[f"{prefix}.ln_2.weight"] = hf_state[f"{hf_prefix}.ln_2.weight"]
        minigpt_state[f"{prefix}.ln_2.bias"] = hf_state[f"{hf_prefix}.ln_2.bias"]
    
    return minigpt_state


def export_to_hf(ckpt_path, output_dir, push_to_hub=False, repo_id=None, token=None):
    """Export mini-GPT checkpoint to HF format."""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Extract config
    minigpt_config = {
        "vocab_size": ckpt.get("vocab_size", 65),
        "max_len": ckpt.get("max_len", 128),
        "d_model": ckpt.get("d_model", 384),
        "num_layers": ckpt.get("num_layers", 6),
        "num_heads": ckpt.get("num_heads", 6),
        "d_ff": ckpt.get("d_ff", 1536),
        "rope": ckpt.get("rope", True),
        "num_kv_heads": ckpt.get("num_kv_heads"),
        "tie_weights": ckpt.get("tie_weights", True),
    }
    
    # Create HF config
    hf_config = create_hf_config(minigpt_config)
    
    # Convert state dict
    hf_state = convert_state_dict_to_hf(ckpt["model"], hf_config)
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    
    # Save model weights (try safetensors first)
    try:
        from safetensors.torch import save_file
        save_file(hf_state, output_dir / "model.safetensors")
        print("Saved model.safetensors")
    except ImportError:
        torch.save(hf_state, output_dir / "pytorch_model.bin")
        print("Saved pytorch_model.bin (install safetensors for .safetensors format)")
    
    # Save tokenizer if available
    if "tokenizer" in ckpt:
        # Would need to implement tokenizer saving
        pass
    
    print(f"Exported to {output_dir}")
    
    # Push to hub
    if push_to_hub and repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Pushed to https://huggingface.co/{repo_id}")
        except ImportError:
            print("huggingface_hub not installed. Run: pip install huggingface_hub")


def import_from_hf(hf_path, output_path, vocab_size=None, max_len=None):
    """Import HF model to mini-GPT format."""
    print(f"Loading HF model from {hf_path}...")
    
    # Load config
    with open(Path(hf_path) / "config.json", "r") as f:
        hf_config = json.load(f)
    
    # Load weights
    try:
        from safetensors.torch import load_file
        hf_state = load_file(Path(hf_path) / "model.safetensors")
    except (ImportError, FileNotFoundError):
        hf_state = torch.load(Path(hf_path) / "pytorch_model.bin", map_location="cpu", weights_only=False)
    
    # Convert
    minigpt_state = convert_state_dict_from_hf(hf_state, hf_config)
    
    # Build mini-GPT config
    minigpt_config = {
        "vocab_size": vocab_size or hf_config.get("vocab_size", 50257),
        "max_len": max_len or hf_config.get("n_positions", 1024),
        "d_model": hf_config.get("n_embd", 768),
        "num_layers": hf_config.get("n_layer", 12),
        "num_heads": hf_config.get("n_head", 12),
        "d_ff": hf_config.get("n_inner", 3072),
        "rope": hf_config.get("rope", False),
        "num_kv_heads": hf_config.get("num_kv_heads"),
        "tie_weights": hf_config.get("tie_weights", True),
    }
    
    # Save
    torch.save({
        "model": minigpt_state,
        **minigpt_config
    }, output_path)
    
    print(f"Saved mini-GPT checkpoint to {output_path}")


def test_hf_model(hf_path, prompt="To be, or not to be"):
    """Test HF model with transformers library."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed. Run: pip install transformers")
        return
    
    print(f"Loading HF model from {hf_path}...")
    model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")


def main():
    parser = argparse.ArgumentParser(description="Convert mini-GPT <-> Hugging Face")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Export
    export_parser = subparsers.add_parser("export", help="Export to HF format")
    export_parser.add_argument("--ckpt", type=str, required=True, help="mini-GPT checkpoint")
    export_parser.add_argument("--output", type=str, required=True, help="Output directory")
    export_parser.add_argument("--push-to-hub", action="store_true", help="Push to HF Hub")
    export_parser.add_argument("--repo-id", type=str, help="HF Hub repo ID")
    export_parser.add_argument("--token", type=str, help="HF token")
    
    # Import
    import_parser = subparsers.add_parser("import", help="Import from HF format")
    import_parser.add_argument("--hf-path", type=str, required=True, help="HF model path")
    import_parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    import_parser.add_argument("--vocab-size", type=int, help="Override vocab size")
    import_parser.add_argument("--max-len", type=int, help="Override max length")
    
    # Test
    test_parser = subparsers.add_parser("test", help="Test HF model")
    test_parser.add_argument("--hf-path", type=str, required=True, help="HF model path")
    test_parser.add_argument("--prompt", type=str, default="To be, or not to be", help="Test prompt")
    
    args = parser.parse_args()
    
    if args.command == "export":
        export_to_hf(args.ckpt, args.output, args.push_to_hub, args.repo_id, args.token)
    elif args.command == "import":
        import_from_hf(args.hf_path, args.output, args.vocab_size, args.max_len)
    elif args.command == "test":
        test_hf_model(args.hf_path, args.prompt)


if __name__ == "__main__":
    main()