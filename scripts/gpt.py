"""
Mini-GPT CLI: train and sample a character-level language model.

Usage:
  python scripts/gpt.py --epochs 30 --save
  python scripts/gpt.py --load --sample
  python scripts/gpt.py --load --rope --load-path checkpoints/gpt_rope_gqa.pt --top-p 0.9

The model implementation lives in src/transformer/gpt.py; this script is
only the command-line interface.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

import torch  # noqa: E402

from transformer.gpt import GPT, CharTokenizer, train, sample  # noqa: E402
# CharTokenizer must be in THIS namespace: checkpoints pickle it as
# `__main__.CharTokenizer`, so torch.load resolves the class against the
# importing module — removing this import breaks loading.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--load", action="store_true", help="load a checkpoint and sample")
    parser.add_argument("--save", action="store_true", help="save best checkpoint")
    parser.add_argument("--prompt", type=str, default="To be, or not to be")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n-chars", type=int, default=400)
    parser.add_argument("--rope", action="store_true", help="use rotary positional embeddings")
    parser.add_argument("--load-path", type=str, default=os.path.join(CKPT, "gpt.pt"),
                        help="checkpoint to load when --load is set")
    parser.add_argument("--save-path", type=str, default=os.path.join(CKPT, "gpt.pt"),
                        help="checkpoint to write when --save is set")
    parser.add_argument("--kv-heads", type=int, default=None,
                        help="K/V heads per layer (GQA; must divide --heads). Default: one per query head")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="nucleus sampling mass (default 1.0 = plain temperature sampling)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed RNGs for reproducible training")
    args = parser.parse_args()

    if args.load:
        ckpt = torch.load(args.load_path, map_location="cpu", weights_only=False)
        tokenizer = ckpt["tokenizer"]
        has_learned_pos = "pos_embedding.weight" in ckpt["model"]
        if args.rope and has_learned_pos:
            raise SystemExit(
                f"{args.load_path} was trained with LEARNED position embeddings — "
                f"drop --rope to load it."
            )
        if not args.rope and not has_learned_pos:
            raise SystemExit(
                f"{args.load_path} was trained with RoPE — pass --rope to load it."
            )
        num_kv_heads = args.kv_heads or ckpt.get("num_kv_heads")
        model = GPT(vocab_size=tokenizer.vocab_size, max_len=args.block_size,
                    rope=args.rope, num_kv_heads=num_kv_heads)
        model.load_state_dict(ckpt["model"])
        sample(model, tokenizer, args.prompt, args.n_chars, args.temperature,
               args.top_p)
    else:
        with open(os.path.join(ROOT, "data", "shakespeare.txt"), "r", encoding="utf-8") as f:
            text = f.read()

        tokenizer = CharTokenizer(text)
        print(f"Vocab size: {tokenizer.vocab_size} chars")

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n_val = int(0.1 * len(data))
        train_data, val_data = data[:-n_val], data[-n_val:]

        t0 = time.time()
        model = train(
            tokenizer, train_data, val_data,
            epochs=args.epochs, block_size=args.block_size,
            d_model=args.d_model, num_heads=args.heads, d_ff=args.d_ff,
            num_layers=args.layers, batch_size=args.batch_size, save=args.save,
            rope=args.rope, save_path=args.save_path, num_kv_heads=args.kv_heads,
            seed=args.seed,
        )
        print(f"Trained in {time.time() - t0:.1f}s\n")
        sample(model, tokenizer, args.prompt, args.n_chars, args.temperature)