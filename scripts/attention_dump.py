"""
Print attention weights as text matrices so patterns can be inspected in the terminal.

Usage: python scripts/attention_dump.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

from transformer.model import create_masks  # noqa: E402
from visualize import load_model, SOS_TOKEN, PAD_TOKEN  # noqa: E402


def fmt(w: torch.Tensor) -> str:
    return "\n".join(
        "  ".join(f"{v:5.2f}" for v in row) for row in w.tolist()
    )


def main():
    model = load_model("reverse")
    src = torch.tensor([[9, 15, 3, 8, 5]])
    tgt_input = torch.tensor([[SOS_TOKEN, 5, 8, 3, 15]])
    src_mask, tgt_mask = create_masks(src, tgt_input, PAD_TOKEN)

    with torch.no_grad():
        enc_out = model.encoder(src, src_mask)
        model.decoder(tgt_input, enc_out, tgt_mask, src_mask)

    src_tokens = [str(t) for t in src[0].tolist()]
    tgt_tokens = [str(t) for t in tgt_input[0].tolist()]

    for layer_idx, layer in enumerate(model.encoder.layers):
        print(f"=== Encoder self-attention, layer {layer_idx} (head 0) ===")
        print("  ", "  ".join(f"{t:>3}" for t in src_tokens))
        w = layer.self_attn.last_attn_weights[0][0]
        for i, row in enumerate(w):
            print(f"{src_tokens[i]:>2} " + "  ".join(f"{v:5.2f}" for v in row))
        print()

    for layer_idx, layer in enumerate(model.decoder.layers):
        print(f"=== Decoder cross-attention, layer {layer_idx} (head 0) ===")
        # Rows = queries (target tokens), columns = keys (source tokens)
        print("       " + "  ".join(f"k={t:>3}" for t in src_tokens))
        w = layer.cross_attn.last_attn_weights[0][0]
        for i, row in enumerate(w):
            print(f"q={tgt_tokens[i]:>3} " + "  ".join(f"{v:5.2f}" for v in row))
        print()


if __name__ == "__main__":
    main()