"""
Train the seq2seq Transformer (reverse/copy tasks).

Usage:
  python scripts/train_seq2seq.py --task reverse --epochs 100
  python scripts/train_seq2seq.py --task copy --epochs 100

Saves the trained model to checkpoints/model.pt so scripts/visualize.py
and scripts/attention_dump.py can load it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from transformer.trainer import train, evaluate, DATASETS  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(DATASETS), default="reverse")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    model, device = train(
        args.task, args.epochs,
        save_path=os.path.join(ROOT, "checkpoints", "model.pt"),
    )
    evaluate(model, device, args.task)