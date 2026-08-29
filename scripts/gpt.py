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

from transformer.gpt import GPT, CharTokenizer, train, sample, beam_sample  # noqa: E402
from transformer.bpe import BPETokenizer  # noqa: E402
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
    parser.add_argument("--amp", action="store_true",
                        help="train with fp16 autocast + GradScaler (CUDA/MPS)")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="accumulate gradients over this many batches before stepping")
    parser.add_argument("--compile", action="store_true",
                        help="wrap the model with torch.compile")
    parser.add_argument("--tb-log", type=str, default=None,
                        help="log scalars to a TensorBoard dir (needs `pip install tensorboard`)")
    parser.add_argument("--bpe", action="store_true",
                        help="use a byte-level BPE tokenizer instead of character-level")
    parser.add_argument("--bpe-vocab-size", type=int, default=512,
                        help="BPE vocabulary size (256 base bytes + merges)")
    parser.add_argument("--tie-weights", action="store_true",
                        help="share the token embedding and output head (GPT-2 style)")
    parser.add_argument("--num-pairs", type=int, default=20000,
                        help="training slices per epoch (bigger = more data per epoch)")
    parser.add_argument("--num-val-pairs", type=int, default=2000,
                        help="validation slices per epoch")
    parser.add_argument("--beam", type=int, default=0,
                        help="beam search width (0 = greedy sampling)")
    parser.add_argument("--length-penalty", type=float, default=0.0,
                        help="length penalty for beam search (>0 penalizes long sequences)")
    parser.add_argument("--rope-scaling", choices=["none", "linear", "ntk"], default="none",
                        help="RoPE scaling for longer context: linear or NTK-aware")
    parser.add_argument("--rope-scaling-factor", type=float, default=1.0,
                        help="scaling factor for RoPE (>1 extends context window)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Adam weight decay (L2 regularization)")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="gradient clipping max norm (0 = disabled)")
    parser.add_argument("--scheduler", choices=["noam", "cosine_restarts"], default="noam",
                        help="learning rate scheduler: noam (Transformer paper) or cosine_restarts")
    parser.add_argument("--scheduler-t0", type=int, default=1000,
                        help="CosineWarmRestarts: steps in first restart period (T_0)")
    parser.add_argument("--scheduler-t-mult", type=int, default=2,
                        help="CosineWarmRestarts: period multiplier after each restart")
    parser.add_argument("--scheduler-eta-min", type=float, default=0.0,
                        help="CosineWarmRestarts: minimum learning rate")
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
        # torch.save preserves tensor sharing: a tied checkpoint's embedding and
        # lm_head entries are the SAME tensor → reconstruct the tying on load.
        tied = ("lm_head.weight" in ckpt["model"] and "token_embedding.weight" in ckpt["model"]
                and ckpt["model"]["lm_head.weight"].data_ptr()
                == ckpt["model"]["token_embedding.weight"].data_ptr())
        model = GPT(vocab_size=tokenizer.vocab_size, max_len=args.block_size,
                    rope=args.rope, num_kv_heads=num_kv_heads, tie_weights=tied,
                    rope_scaling=args.rope_scaling, rope_scaling_factor=args.rope_scaling_factor)
        model.load_state_dict(ckpt["model"])
        if args.beam > 0:
            beam_sample(model, tokenizer, args.prompt, args.n_chars,
                        args.beam, args.length_penalty)
        else:
            sample(model, tokenizer, args.prompt, args.n_chars, args.temperature,
                   args.top_p)
    else:
        with open(os.path.join(ROOT, "data", "shakespeare.txt"), "r", encoding="utf-8") as f:
            text = f.read()

        if args.bpe:
            tokenizer = BPETokenizer(text, vocab_size=args.bpe_vocab_size)
            print(f"BPE vocab size: {tokenizer.vocab_size} "
                  f"({tokenizer.vocab_size - 256} merges over 256 base bytes)")
        else:
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
            seed=args.seed, amp=args.amp, grad_accum=args.grad_accum,
            compile_model=args.compile, tb_log=args.tb_log,
            num_pairs=args.num_pairs, num_val_pairs=args.num_val_pairs,
            tie_weights=args.tie_weights,
            rope_scaling=args.rope_scaling, rope_scaling_factor=args.rope_scaling_factor,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            scheduler=args.scheduler, scheduler_t0=args.scheduler_t0,
            scheduler_t_mult=args.scheduler_t_mult, scheduler_eta_min=args.scheduler_eta_min,
        )
        print(f"Trained in {time.time() - t0:.1f}s\n")
        if args.beam > 0:
            beam_sample(model, tokenizer, args.prompt, args.n_chars,
                        args.beam, args.length_penalty)
        else:
            sample(model, tokenizer, args.prompt, args.n_chars, args.temperature,
                   args.top_p)