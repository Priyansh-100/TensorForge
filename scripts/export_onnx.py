"""
Export the RoPE GPT to ONNX and benchmark ONNX Runtime vs PyTorch.

  pip install onnx onnxruntime
  python scripts/export_onnx.py                     # exports checkpoints/gpt_rope.onnx
  python scripts/export_onnx.py --ckpt checkpoints/gpt_rope_gqa.pt
  python scripts/export_onnx.py --no-bench           # export only

The exported graph uses dynamic batch and sequence axes, so any prompt length
is accepted at inference time.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")

import torch  # noqa: E402
import numpy as np  # noqa: E402

from transformer.gpt import GPT  # noqa: E402
from transformer.model import create_look_ahead_mask  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=os.path.join(CKPT, "gpt_rope.pt"))
    parser.add_argument("--out", type=str, default=os.path.join(CKPT, "gpt_rope.onnx"))
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--n-steps", type=int, default=20)
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401  (ensures the exporter's proto serializer exists)
    except ImportError:
        raise SystemExit("`pip install onnx` first (the torch.onnx exporter needs it)")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tokenizer = ckpt["tokenizer"]
    model = GPT(vocab_size=tokenizer.vocab_size, max_len=128, rope=True,
                num_kv_heads=ckpt.get("num_kv_heads"))
    model.load_state_dict(ckpt["model"])
    model.eval()

    T = 32
    torch.manual_seed(0)
    tokens = torch.randint(0, tokenizer.vocab_size, (1, T), dtype=torch.long)
    mask = create_look_ahead_mask(T)

    with torch.no_grad():
        ref = model(tokens, mask)

    dynamic = {
        "tokens": {0: "batch", 1: "seq"},
        "mask": {0: "batch", 2: "seq", 3: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }
    torch.onnx.export(
        model,
        (tokens, mask),
        args.out,
        input_names=["tokens", "mask"],
        output_names=["logits"],
        dynamic_axes=dynamic,
        opset_version=17,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"exported {args.out} ({size_kb:.0f} KiB, dynamic batch/seq)")

    if args.no_bench:
        return

    try:
        import onnxruntime as ort
    except ImportError:
        print("`pip install onnxruntime` to run the ORT benchmark")
        return

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"tokens": tokens.numpy(), "mask": mask.numpy()})[0]
    max_dev = np.abs(ref.numpy() - got).max().item()
    print(f"  ORT vs torch logits: max |Δ| = {max_dev:.2e} "
          f"{'OK' if max_dev < 1e-4 else 'FAIL'}")

    # prefill throughput on a fixed 128-token prompt
    T2 = 128
    tok2 = torch.randint(0, tokenizer.vocab_size, (1, T2), dtype=torch.long)
    mask2 = create_look_ahead_mask(T2)

    t0 = time.time()
    for _ in range(args.n_steps):
        with torch.no_grad():
            model(tok2, mask2)
    torch_dt = (time.time() - t0) / args.n_steps

    t0 = time.time()
    for _ in range(args.n_steps):
        sess.run(None, {"tokens": tok2.numpy(), "mask": mask2.numpy()})
    ort_dt = (time.time() - t0) / args.n_steps

    print(f"  prefill {T2} tokens: torch {torch_dt * 1000:6.1f} ms "
          f"({T2 / torch_dt:8,.0f} tok/s) | ORT {ort_dt * 1000:6.1f} ms "
          f"({T2 / ort_dt:8,.0f} tok/s)")


if __name__ == "__main__":
    main()