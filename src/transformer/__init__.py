"""
Transformer from scratch — library package.

Modules:
    model      — architecture (scaled dot-product attention, RoPE, MHA, masks,
                 positional encoding, Noam schedule, encoder/decoder).
    gpt        — decoder-only mini-GPT: CharTokenizer, CharDataset, GPT with
                 learned/RoPE positions, GQA, KV-cache generation, training.
    attention  — attention math verified from first principles (manual
                 forward/backward vs autograd, softmax saturation demo).
    trainer    — seq2seq datasets (reverse/copy), training loop, greedy decode.
"""

from transformer.model import (
    MultiHeadAttention,
    FeedForward,
    NoamSchedule,
    PositionalEncoding,
    Transformer,
    Encoder,
    Decoder,
    create_pad_mask,
    create_look_ahead_mask,
    create_masks,
    precompute_rope,
    apply_rope,
    rotate_half,
    scaled_dot_product_attention,
)

from transformer.gpt import (
    GPT,
    GPTBlock,
    CharTokenizer,
    CharDataset,
    _sample_token,
)

# Compatibility: checkpoints trained under the old flat layout pickle the
# tokenizer class with module name "gpt" (gpt.py as a top-level module).
# Keep that name resolvable so torch.load finds CharTokenizer everywhere.
import sys as _sys
from transformer import gpt as _gpt_module

_sys.modules.setdefault("gpt", _gpt_module)
del _sys, _gpt_module

from transformer.trainer import (
    Seq2SeqDataset,
    ReverseDataset,
    CopyDataset,
    DATASETS,
    SOS_TOKEN,
    PAD_TOKEN,
    train,
    greedy_decode,
    evaluate,
)

__all__ = [
    "MultiHeadAttention",
    "FeedForward",
    "NoamSchedule",
    "PositionalEncoding",
    "Transformer",
    "Encoder",
    "Decoder",
    "create_pad_mask",
    "create_look_ahead_mask",
    "create_masks",
    "precompute_rope",
    "apply_rope",
    "rotate_half",
    "scaled_dot_product_attention",
    "GPT",
    "GPTBlock",
    "CharTokenizer",
    "CharDataset",
    "_sample_token",
    "Seq2SeqDataset",
    "ReverseDataset",
    "CopyDataset",
    "DATASETS",
    "SOS_TOKEN",
    "PAD_TOKEN",
    "train",
    "greedy_decode",
    "evaluate",
]
