"""
Visualize attention weights of a trained model.

Usage:
  python train.py --task reverse     # first, train the model (saves model.pt)
  python visualize.py --task reverse # then visualize attention

Produces PNG heatmaps in the "plots/" directory:
  - encoder self-attention (all layers)
  - decoder masked self-attention (all layers)
  - decoder cross-attention (all layers)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformer import Transformer, create_masks

PAD_TOKEN = 0
SOS_TOKEN = 1
NUM_HEADS = 4


def load_model(task: str) -> Transformer:
    model = Transformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=64,
        num_heads=NUM_HEADS,
        d_ff=128,
        num_layers=2,
        max_len=6,
    )
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model


def plot_heatmap(ax, weights: torch.Tensor, x_labels, y_labels, title: str):
    """weights: (q_len, k_len)"""
    im = ax.imshow(weights, cmap="viridis")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main(task: str):
    os.makedirs("plots", exist_ok=True)
    model = load_model(task)

    # A single test example (batch of 1)
    src = torch.tensor([[9, 15, 3, 8, 5]])  # (1, 5)
    tgt_input = torch.tensor([[SOS_TOKEN, 5, 8, 3, 15]])  # SOS + expected target[:-1]
    src_mask, tgt_mask = create_masks(src, tgt_input, PAD_TOKEN)

    with torch.no_grad():
        enc_out = model.encoder(src, src_mask)
        model.decoder(tgt_input, enc_out, tgt_mask, src_mask)

    src_tokens = [str(t) for t in src[0].tolist()]
    tgt_tokens = [str(t) for t in tgt_input[0].tolist()]

    # --- Encoder self-attention ---
    for layer_idx, layer in enumerate(model.encoder.layers):
        w = layer.self_attn.last_attn_weights[0]  # (heads, q_len, k_len)
        fig, axes = plt.subplots(1, NUM_HEADS, figsize=(16, 4))
        for head in range(NUM_HEADS):
            plot_heatmap(
                axes[head],
                w[head].numpy(),
                src_tokens,
                src_tokens,
                f"Layer {layer_idx} | Head {head}",
            )
        fig.suptitle(f"Encoder self-attention — layer {layer_idx} (queries=keys=source)")
        fig.tight_layout()
        fig.savefig(f"plots/encoder_self_attn_layer{layer_idx}.png", dpi=150)
        plt.close(fig)

    # --- Decoder masked self-attention ---
    for layer_idx, layer in enumerate(model.decoder.layers):
        w = layer.self_attn.last_attn_weights[0]  # (heads, q_len, k_len)
        fig, axes = plt.subplots(1, NUM_HEADS, figsize=(16, 4))
        for head in range(NUM_HEADS):
            plot_heatmap(
                axes[head],
                w[head].numpy(),
                tgt_tokens,
                tgt_tokens,
                f"Layer {layer_idx} | Head {head}",
            )
        fig.suptitle(f"Decoder masked self-attention — layer {layer_idx} (should be lower-triangular)")
        fig.tight_layout()
        fig.savefig(f"plots/decoder_self_attn_layer{layer_idx}.png", dpi=150)
        plt.close(fig)

    # --- Decoder cross-attention ---
    for layer_idx, layer in enumerate(model.decoder.layers):
        w = layer.cross_attn.last_attn_weights[0]  # (heads, q_len, k_len)
        fig, axes = plt.subplots(1, NUM_HEADS, figsize=(16, 4))
        for head in range(NUM_HEADS):
            plot_heatmap(
                axes[head],
                w[head].numpy(),
                src_tokens,
                tgt_tokens,
                f"Layer {layer_idx} | Head {head}",
            )
        fig.suptitle(f"Decoder cross-attention — layer {layer_idx} (queries=target, keys=source)")
        fig.tight_layout()
        fig.savefig(f"plots/decoder_cross_attn_layer{layer_idx}.png", dpi=150)
        plt.close(fig)

    print("Saved plots to plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["reverse", "copy"], default="reverse")
    args = parser.parse_args()
    main(args.task)
