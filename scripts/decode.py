"""
Beam search vs greedy decoding for the seq2seq Transformer.

Greedy: at each step take the argmax token. One wrong early choice is unrecoverable.
Beam:   keep the top-k *complete sequences* at every step; only expand those.
        Trade-off: more beams = better search, exponentially more compute (k forward passes).

Usage:
  python scripts/train_seq2seq.py --task reverse --epochs 100   # train first (saves checkpoints/model.pt)
  python scripts/decode.py --task reverse --beam-size 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from transformer.model import Transformer, create_look_ahead_mask  # noqa: E402
from transformer.trainer import greedy_decode, PAD_TOKEN, SOS_TOKEN  # noqa: E402


def beam_decode(
    model: Transformer,
    src: torch.Tensor,           # (1, seq_len)
    beam_size: int,
    max_len: int,
    device,
    length_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Returns the single best sequence.

    State: (seq_len, beam_size) of sequences, each with a log-probability score.
    At each step: expand every beam by every vocab token, keep the best beam_size.
    """
    model.eval()
    src = src.to(device)
    src_mask = (src != PAD_TOKEN).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)

    vocab_size = model.output_proj.out_features

    # Beams: (1, 1, 1) → expand to (beam, 1) — all beams start at [SOS]
    beams = torch.full((beam_size, 1), SOS_TOKEN, dtype=torch.long, device=device)
    scores = torch.zeros(beam_size, device=device)  # cumulative log-prob

    for _ in range(max_len):
        # Encode all beams in one batched forward pass
        look_ahead = create_look_ahead_mask(beams.size(1)).to(device)
        tgt_mask = look_ahead  # no padding tokens in beams

        with torch.no_grad():
            dec_output = model.decoder(beams, enc_output, tgt_mask, src_mask)
            logits = model.output_proj(dec_output)  # (beam, step, vocab)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (beam, vocab)

        # Combine: beam score + token score, flatten, take top-k
        # (beam, vocab) → (beam * vocab) candidates
        flat_scores = (scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_idx = flat_scores.topk(min(beam_size, flat_scores.size(0)))

        # Decompose flattened index → (beam, vocab) coordinates
        beam_idx = top_idx // vocab_size
        token_idx = top_idx % vocab_size

        # Build new beams: previous sequence + new token
        new_beams = torch.cat([beams[beam_idx], token_idx.unsqueeze(1)], dim=1)
        beams, scores = new_beams, top_scores

    # Pick the best final beam; normalize by length so longer sequences aren't punished
    norm_scores = scores / (beams.size(1) ** length_penalty) if length_penalty > 0 else scores
    best = beams[norm_scores.argmax()]
    return best[1:]  # strip SOS


def compare(task: str, beam_size: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=20, tgt_vocab_size=20,
        d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=6,
    )
    model.load_state_dict(torch.load(os.path.join(CKPT, "model.pt"), map_location="cpu"))
    model.to(device).eval()

    vocab_size = 20
    seq_len = 5
    test_src = torch.randint(2, vocab_size, (8, seq_len))

    correct = {"greedy": 0, f"beam{beam_size}": 0}
    for i in range(test_src.size(0)):
        src_seq = test_src[i].tolist()
        expected = src_seq[::-1] if task == "reverse" else src_seq[:]

        g = greedy_decode(model, test_src[i].unsqueeze(0), device, max_len=seq_len)
        b = beam_decode(model, test_src[i].unsqueeze(0), beam_size, seq_len, device)

        if g[0].tolist() == expected:
            correct["greedy"] += 1
        if b.tolist() == expected:
            correct[f"beam{beam_size}"] += 1

    print(f"Accuracy on {test_src.size(0)} examples")
    for name, n in correct.items():
        print(f"  {name:>8}: {n}/{test_src.size(0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["reverse", "copy"], default="reverse")
    parser.add_argument("--beam-size", type=int, default=4)
    args = parser.parse_args()
    compare(args.task, args.beam_size)